package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"text/template"

	"github.com/invopop/jsonschema"
	"github.com/mattn/go-sqlite3"
	"github.com/openai/openai-go"
	"github.com/reeflective/readline"
)

type Column struct {
	Name        string `json:"name" jsonschema_description:"The name of the column, lowercase, no spaces"`
	Type        string `json:"type" jsonschema_description:"The SQLite type of the column" jsonschema:"enum=INTEGER,enum=TEXT,enum=REAL,enum=BLOB"`
	Constraints string `json:"constraints" jsonschema_description:"SQL constraints for the column (e.g., PRIMARY KEY, UNIQUE)"`
	Description string `json:"description" jsonschema_description:"A brief description of the column"`
}

type TableSchema struct {
	Columns []Column `json:"columns" jsonschema_description:"The list of columns for the table"`
}

var schemaCache = make(map[string][]Column)

type PromptData struct {
	TableName string
	TableDesc string
}

func renderPrompt(templatePath string, data interface{}) (string, error) {
	tmpl, err := template.ParseFiles(templatePath)
	if err != nil {
		return "", fmt.Errorf("failed to parse template %s: %w", templatePath, err)
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("failed to execute template %s: %w", templatePath, err)
	}
	return buf.String(), nil
}

type InfiniDBModule struct{}

func (m *InfiniDBModule) Create(c *sqlite3.SQLiteConn, args []string) (sqlite3.VTab, error) {
	return m.Connect(c, args)
}

func (m *InfiniDBModule) Connect(c *sqlite3.SQLiteConn, args []string) (sqlite3.VTab, error) {
	//fmt.Println("Connect called with args:", args)
	if len(args) < 4 {
		return nil, fmt.Errorf("missing table name argument")
	}
	tableName := args[2]
	tableDesc := args[3]
	//fmt.Println("Creating table:", tableName)
	//fmt.Println("Description:", tableDesc)

	var columns []Column
	if cached, ok := schemaCache[tableName]; ok {
		columns = cached
	} else {
		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("OPENAI_API_KEY not set")
		}
		client := openai.NewClient()
		
		schemaParam := openai.ResponseFormatJSONSchemaJSONSchemaParam{
			Name:        "table_schema",
			Description: openai.String("Schema definition for a SQLite table"),
			Schema:      GenerateSchema[TableSchema](),
			Strict:      openai.Bool(true),
		}

		prompt, err := renderPrompt("prompts/schema_generation.txt", PromptData{TableName: tableName, TableDesc: tableDesc})
		if err != nil {
			return nil, fmt.Errorf("failed to render schema prompt: %w", err)
		}

		chat, err := client.Chat.Completions.New(context.Background(), openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(prompt),
			},
			ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{JSONSchema: schemaParam},
			},
			Model: openai.ChatModelGPT4o2024_08_06,
		})

		if err != nil {
			return nil, fmt.Errorf("schema generation failed: %w", err)
		}

		var tableSchema TableSchema
		if err := json.Unmarshal([]byte(chat.Choices[0].Message.Content), &tableSchema); err != nil {
			return nil, fmt.Errorf("failed to parse schema JSON: %w", err)
		}
		columns = tableSchema.Columns

		if len(columns) == 0 {
			return nil, fmt.Errorf("no columns generated")
		}

		// Validate columns
		seenNames := make(map[string]bool)
		validTypes := map[string]bool{"INTEGER": true, "TEXT": true, "REAL": true, "BLOB": true}
		for _, col := range columns {
			if col.Name == "" || seenNames[col.Name] {
				return nil, fmt.Errorf("invalid or duplicate column name: %s", col.Name)
			}
			seenNames[col.Name] = true
			if !validTypes[col.Type] {
				return nil, fmt.Errorf("invalid column type: %s for %s", col.Type, col.Name)
			}
		}

		schemaCache[tableName] = columns
	}

	// Build schema string
	schemaParts := []string{}
	for _, col := range columns {
		part := fmt.Sprintf("%s %s", col.Name, col.Type)
		if col.Constraints != "" {
			part += " " + col.Constraints
		}
		schemaParts = append(schemaParts, part)
	}
	schema := "CREATE TABLE virtual_table (" + strings.Join(schemaParts, ", ") + ")"

	if err := c.DeclareVTab(schema); err != nil {
		return nil, err
	}

	return &InfiniTable{tableName: tableName, tableDesc: tableDesc, columns: columns}, nil
}

func (m *InfiniDBModule) DestroyModule() {}

type InfiniTable struct {
	tableName string
	tableDesc string
	columns   []Column
}

func (vt *InfiniTable) BestIndex(cst []sqlite3.InfoConstraint, ob []sqlite3.InfoOrderBy) (*sqlite3.IndexResult, error) {
	return &sqlite3.IndexResult{
		Used: make([]bool, len(cst)),
	}, nil
}

func (vt *InfiniTable) Disconnect() error { return nil }
func (vt *InfiniTable) Destroy() error    { return nil }

func (vt *InfiniTable) Open() (sqlite3.VTabCursor, error) {
	cacheDir := ".cache"
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		fmt.Printf("Warning: failed to create cache directory: %v\n", err)
	}

	cacheFile := filepath.Join(cacheDir, fmt.Sprintf("%s_data.json", vt.tableName))
	if data, err := os.ReadFile(cacheFile); err == nil {
		var rows []map[string]interface{}
		if err := json.Unmarshal(data, &rows); err == nil {
			fmt.Println("Loading data from cache for table:", vt.tableName)
			return &InfiniCursor{tableName: vt.tableName, data: rows, pos: 0, columns: vt.columns}, nil
		}
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY not set")
	}
	client := openai.NewClient()

	dataSchema := makeDataSchema(vt.columns)
	schemaParam := openai.ResponseFormatJSONSchemaJSONSchemaParam{
		Name:        "table_data",
		Description: openai.String("Generated data rows"),
		Schema:      dataSchema,
		Strict:      openai.Bool(true),
	}

	prompt, err := renderPrompt("prompts/data_generation.txt", PromptData{TableName: vt.tableName, TableDesc: vt.tableDesc})
	if err != nil {
		return nil, fmt.Errorf("failed to render data prompt: %w", err)
	}

	chat, err := client.Chat.Completions.New(context.Background(), openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(prompt),
		},
		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{JSONSchema: schemaParam},
		},
		Model: openai.ChatModelGPT4o2024_08_06,
	})

	if err != nil {
		return nil, fmt.Errorf("data generation failed: %w", err)
	}

	// We need a struct to unmarshal the response which has a "rows" field
	type DataResponse struct {
		Rows []map[string]interface{} `json:"rows"`
	}
	var dataResp DataResponse
	if err := json.Unmarshal([]byte(chat.Choices[0].Message.Content), &dataResp); err != nil {
		return nil, fmt.Errorf("failed to parse data JSON: %w", err)
	}

	if len(dataResp.Rows) == 0 {
		return nil, fmt.Errorf("no data generated")
	}

	if jsonData, err := json.Marshal(dataResp.Rows); err == nil {
		if err := os.WriteFile(cacheFile, jsonData, 0644); err != nil {
			fmt.Printf("Warning: failed to write to cache: %v\n", err)
		}
	}

	return &InfiniCursor{tableName: vt.tableName, data: dataResp.Rows, pos: 0, columns: vt.columns}, nil
}

type InfiniCursor struct {
	tableName string
	data      []map[string]interface{}
	pos       int
	columns   []Column
}

func (cur *InfiniCursor) Close() error { return nil }
func (cur *InfiniCursor) Filter(idxNum int, idxStr string, vals []interface{}) error {
	cur.pos = 0
	return nil
}
func (cur *InfiniCursor) Next() error {
	cur.pos += 1
	return nil
}

func (cur *InfiniCursor) EOF() bool {
	return cur.pos >= len(cur.data)
}

func (cur *InfiniCursor) Rowid() (int64, error) {
	return int64(cur.pos), nil
}

func (cur *InfiniCursor) Column(c *sqlite3.SQLiteContext, col int) error {
	if cur.pos < 0 || cur.pos >= len(cur.data) || col < 0 || col >= len(cur.columns) {
		return fmt.Errorf("invalid cursor position or column index")
	}

	row := cur.data[cur.pos]
	colName := cur.columns[col].Name
	value, ok := row[colName]
	if !ok {
		c.ResultNull()
		return nil
	}

	switch cur.columns[col].Type {
	case "INTEGER":
		if v, ok := value.(float64); ok {
			c.ResultInt64(int64(v))
		} else if v, ok := value.(int64); ok {
			c.ResultInt64(v)
		} else {
			// Try parsing string if needed, but structured output should be correct type
			return fmt.Errorf("type mismatch for %s: expected integer", colName)
		}
	case "TEXT":
		if v, ok := value.(string); ok {
			c.ResultText(v)
		} else {
			return fmt.Errorf("type mismatch for %s: expected text", colName)
		}
	case "REAL":
		if v, ok := value.(float64); ok {
			c.ResultDouble(v)
		} else {
			return fmt.Errorf("type mismatch for %s: expected real", colName)
		}
	case "BLOB":
		if v, ok := value.(string); ok {
			c.ResultBlob([]byte(v))
		} else if v, ok := value.([]byte); ok {
			c.ResultBlob(v)
		} else {
			return fmt.Errorf("type mismatch for %s: expected blob", colName)
		}
	default:
		return fmt.Errorf("unsupported column type: %s", cur.columns[col].Type)
	}
	return nil
}

func GenerateSchema[T any]() interface{} {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T
	schema := reflector.Reflect(v)
	return schema
}

func makeDataSchema(columns []Column) interface{} {
	properties := make(map[string]interface{})
	required := []string{}

	for _, col := range columns {
		colSchema := map[string]interface{}{}
		switch col.Type {
		case "INTEGER":
			colSchema["type"] = "integer"
		case "REAL":
			colSchema["type"] = "number"
		case "TEXT", "BLOB":
			colSchema["type"] = "string"
		default:
			colSchema["type"] = "string"
		}
		colSchema["description"] = col.Description
		properties[col.Name] = colSchema
		required = append(required, col.Name)
	}

	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"rows": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "object",
					"properties": properties,
					"required": required,
					"additionalProperties": false,
				},
			},
		},
		"required": []string{"rows"},
		"additionalProperties": false,
	}
	return schema
}

func init() {
	sql.Register("infinidb",
		&sqlite3.SQLiteDriver{
			ConnectHook: func(conn *sqlite3.SQLiteConn) error {
				return conn.CreateModule("infinidb", &InfiniDBModule{})
			},
		})
}

func main() {
	db, err := sql.Open("infinidb", ":memory:")
	if err != nil {
		panic(err)
	}
	defer db.Close()

	fmt.Println("Welcome to InfiniDB REPL")
	fmt.Println("Type 'exit' or 'quit' to leave.")

	rl := readline.NewShell()
	rl.Prompt.Primary(func() string { return "> " })

	for {
		input, err := rl.Readline()
		if err != nil {
			break
		}
		input = strings.TrimSpace(input)
		if input == "" {
			continue
		}
		if strings.EqualFold(input, "exit") || strings.EqualFold(input, "quit") {
			break
		}

		upperInput := strings.ToUpper(input)
		if strings.HasPrefix(upperInput, "SELECT") ||
			strings.HasPrefix(upperInput, "PRAGMA") ||
			strings.HasPrefix(upperInput, "EXPLAIN") {

			rows, err := db.Query(input)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				continue
			}

			cols, err := rows.Columns()
			if err != nil {
				fmt.Printf("Error getting columns: %v\n", err)
				rows.Close()
				continue
			}

			w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)

			// Print headers
			for i, col := range cols {
				fmt.Fprintf(w, "%s", col)
				if i < len(cols)-1 {
					fmt.Fprintf(w, "\t")
				}
			}
			fmt.Fprintln(w)

			// Print rows
			for rows.Next() {
				values := make([]interface{}, len(cols))
				valuePtrs := make([]interface{}, len(cols))
				for i := range values {
					valuePtrs[i] = &values[i]
				}

				if err := rows.Scan(valuePtrs...); err != nil {
					fmt.Printf("Scan error: %v\n", err)
					break
				}

				for i, val := range values {
					if b, ok := val.([]byte); ok {
						fmt.Fprintf(w, "%s", string(b))
					} else {
						fmt.Fprintf(w, "%v", val)
					}
					if i < len(cols)-1 {
						fmt.Fprintf(w, "\t")
					}
				}
				fmt.Fprintln(w)
			}
			w.Flush()
			rows.Close()

		} else {
			_, err := db.Exec(input)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
			}
		}
	}
}
