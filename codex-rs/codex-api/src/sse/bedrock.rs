//! Bedrock non-streaming response handler
//!
//! Bedrock's invoke endpoint returns a single JSON response (not SSE).
//! This module converts that response into ResponseEvents for compatibility.

use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use codex_client::StreamResponse;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use futures::StreamExt;
use tokio::sync::mpsc;
use tracing::debug;

/// Spawn a task to handle Bedrock's non-streaming JSON response
pub fn spawn_bedrock_response(stream_response: StreamResponse) -> ResponseStream {
    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(64);
    tokio::spawn(async move {
        process_bedrock_response(stream_response.bytes, tx_event).await;
    });
    ResponseStream { rx_event }
}

async fn process_bedrock_response<S>(
    mut stream: S,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
) where
    S: futures::Stream<Item = Result<bytes::Bytes, codex_client::TransportError>> + Unpin,
{
    // Collect all bytes from the response
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => body.extend_from_slice(&bytes),
            Err(e) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(format!("Error reading response: {}", e))))
                    .await;
                return;
            }
        }
    }

    // Parse as JSON
    let response: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            debug!("Failed to parse Bedrock response: {}, body: {:?}", e, String::from_utf8_lossy(&body));
            let _ = tx_event
                .send(Err(ApiError::Stream(format!("Failed to parse Bedrock response: {}", e))))
                .await;
            return;
        }
    };

    // Check for error response
    if let Some(message) = response.get("message").and_then(|m| m.as_str()) {
        let _ = tx_event
            .send(Err(ApiError::Stream(format!("Bedrock error: {}", message))))
            .await;
        return;
    }

    // Parse Bedrock/Claude response format
    // {
    //   "id": "msg_...",
    //   "type": "message",
    //   "role": "assistant",
    //   "content": [
    //     {"type": "text", "text": "..."},
    //     {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
    //   ],
    //   "stop_reason": "end_turn" | "tool_use" | "max_tokens",
    //   "usage": {"input_tokens": N, "output_tokens": N}
    // }

    let content = response.get("content").and_then(|c| c.as_array());
    let stop_reason = response.get("stop_reason").and_then(|r| r.as_str());

    if let Some(content_items) = content {
        for item in content_items {
            let item_type = item.get("type").and_then(|t| t.as_str());

            match item_type {
                Some("thinking") => {
                    // Extended thinking block
                    if let Some(thinking_text) = item.get("thinking").and_then(|t| t.as_str()) {
                        let reasoning = ResponseItem::Reasoning {
                            id: String::new(),
                            summary: Vec::new(),
                            content: Some(vec![
                                codex_protocol::models::ReasoningItemContent::ReasoningText {
                                    text: thinking_text.to_string(),
                                },
                            ]),
                            encrypted_content: None,
                        };
                        let _ = tx_event
                            .send(Ok(ResponseEvent::OutputItemDone(reasoning)))
                            .await;
                    }
                }
                Some("text") => {
                    // Text content
                    if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                        // Send text delta for streaming display
                        let _ = tx_event
                            .send(Ok(ResponseEvent::OutputTextDelta(text.to_string())))
                            .await;

                        // Send completed message
                        let message = ResponseItem::Message {
                            id: None,
                            role: "assistant".to_string(),
                            content: vec![ContentItem::OutputText {
                                text: text.to_string(),
                            }],
                        };
                        let _ = tx_event
                            .send(Ok(ResponseEvent::OutputItemDone(message)))
                            .await;
                    }
                }
                Some("tool_use") => {
                    // Tool call
                    let name = item
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    let call_id = item
                        .get("id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("")
                        .to_string();
                    let input = item.get("input").cloned().unwrap_or(serde_json::json!({}));
                    let arguments = serde_json::to_string(&input).unwrap_or_default();

                    let tool_call = ResponseItem::FunctionCall {
                        id: None,
                        name,
                        arguments,
                        call_id,
                    };
                    let _ = tx_event
                        .send(Ok(ResponseEvent::OutputItemDone(tool_call)))
                        .await;
                }
                _ => {
                    debug!("Unknown Bedrock content type: {:?}", item_type);
                }
            }
        }
    }

    // Check for max_tokens (context window exceeded)
    if stop_reason == Some("max_tokens") {
        let _ = tx_event.send(Err(ApiError::ContextWindowExceeded)).await;
        return;
    }

    // Send completion event
    let _ = tx_event
        .send(Ok(ResponseEvent::Completed {
            response_id: response
                .get("id")
                .and_then(|i| i.as_str())
                .unwrap_or("")
                .to_string(),
            token_usage: None,
        }))
        .await;
}
