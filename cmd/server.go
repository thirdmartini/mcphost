package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/charmbracelet/log"
	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"

	"github.com/mark3labs/mcphost/pkg/history"
	"github.com/mark3labs/mcphost/pkg/llm"
)

type Request struct {
	Prompt string
}

type Response struct {
	Message string
}

func runServer(ctx context.Context,
	provider llm.Provider,
	mcpClients map[string]mcpclient.MCPClient,
	tools []llm.Tool,
) error {
	messages := make([]history.HistoryMessage, 0)

	http.HandleFunc("/api/v1/chat", func(w http.ResponseWriter, r *http.Request) {
		request := Request{}

		err := json.NewDecoder(r.Body).Decode(&request)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		message, err := runPromptNonInteractive(ctx, provider, mcpClients, tools, request.Prompt, &messages)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}

		if len(message) > 0 {
			messages = pruneMessages(messages)
		}
		json.NewEncoder(w).Encode(Response{Message: message})
	})
	return http.ListenAndServe(":6002", nil)
}

func runPromptNonInteractive(
	ctx context.Context,
	provider llm.Provider,
	mcpClients map[string]mcpclient.MCPClient,
	tools []llm.Tool,
	prompt string,
	messages *[]history.HistoryMessage,
) (string, error) {
	var message llm.Message
	var err error

	// Convert MessageParam to llm.Message for provider
	// Messages already implement llm.Message interface
	llmMessages := make([]llm.Message, len(*messages))
	for i := range *messages {
		llmMessages[i] = &(*messages)[i]
	}

	message, err = provider.CreateMessage(
		ctx,
		prompt,
		llmMessages,
		tools,
	)

	if err != nil {
		return "", err
	}

	var messageContent []history.ContentBlock

	if message.GetContent() != "" {
		return message.GetContent(), nil
	}

	toolResults := []history.ContentBlock{}
	messageContent = []history.ContentBlock{}

	// Add text content
	if message.GetContent() != "" {
		messageContent = append(messageContent, history.ContentBlock{
			Type: "text",
			Text: message.GetContent(),
		})
	}

	// Handle tool calls
	for _, toolCall := range message.GetToolCalls() {
		input, _ := json.Marshal(toolCall.GetArguments())
		messageContent = append(messageContent, history.ContentBlock{
			Type:  "tool_use",
			ID:    toolCall.GetID(),
			Name:  toolCall.GetName(),
			Input: input,
		})

		// Log usage statistics if available
		inputTokens, outputTokens := message.GetUsage()
		if inputTokens > 0 || outputTokens > 0 {
			log.Info("Usage statistics",
				"input_tokens", inputTokens,
				"output_tokens", outputTokens,
				"total_tokens", inputTokens+outputTokens)
		}

		parts := strings.Split(toolCall.GetName(), "__")
		if len(parts) != 2 {
			fmt.Printf(
				"Error: Invalid tool name format: %s\n",
				toolCall.GetName(),
			)
			continue
		}

		serverName, toolName := parts[0], parts[1]
		mcpClient, ok := mcpClients[serverName]
		if !ok {
			fmt.Printf("Error: Server not found: %s\n", serverName)
			continue
		}

		var toolArgs map[string]interface{}
		if err := json.Unmarshal(input, &toolArgs); err != nil {
			fmt.Printf("Error parsing tool arguments: %v\n", err)
			continue
		}

		var toolResultPtr *mcp.CallToolResult
		req := mcp.CallToolRequest{}
		req.Params.Name = toolName
		req.Params.Arguments = toolArgs
		toolResultPtr, err = mcpClient.CallTool(
			context.Background(),
			req,
		)

		if err != nil {
			errMsg := fmt.Sprintf(
				"Error calling tool %s: %v",
				toolName,
				err,
			)
			fmt.Printf("\n%s\n", errorStyle.Render(errMsg))

			// Add error message as tool result
			toolResults = append(toolResults, history.ContentBlock{
				Type:      "tool_result",
				ToolUseID: toolCall.GetID(),
				Content: []history.ContentBlock{{
					Type: "text",
					Text: errMsg,
				}},
			})
			continue
		}

		toolResult := *toolResultPtr

		if toolResult.Content != nil {
			log.Debug("raw tool result content", "content", toolResult.Content)

			// Create the tool result block
			resultBlock := history.ContentBlock{
				Type:      "tool_result",
				ToolUseID: toolCall.GetID(),
				Content:   toolResult.Content,
			}

			// Extract text content
			var resultText string
			// Handle array content directly since we know it's []interface{}
			for _, item := range toolResult.Content {
				if contentMap, ok := item.(mcp.TextContent); ok {
					resultText += fmt.Sprintf("%v ", contentMap.Text)
				}
			}

			resultBlock.Text = strings.TrimSpace(resultText)
			log.Debug("created tool result block",
				"block", resultBlock,
				"tool_id", toolCall.GetID())

			toolResults = append(toolResults, resultBlock)
		}
	}

	*messages = append(*messages, history.HistoryMessage{
		Role:    message.GetRole(),
		Content: messageContent,
	})

	if len(toolResults) > 0 {
		for _, toolResult := range toolResults {
			*messages = append(*messages, history.HistoryMessage{
				Role:    "tool",
				Content: []history.ContentBlock{toolResult},
			})
		}
		// Make another call to get Claude's response to the tool results
		return runPromptNonInteractive(ctx, provider, mcpClients, tools, "", messages)
	}
	return "", nil
}
