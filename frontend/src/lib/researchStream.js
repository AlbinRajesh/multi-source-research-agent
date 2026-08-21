const API_BASE = "http://localhost:8001"; // matches your main.py docstring

export async function streamResearch(topic, threadId, { onNode, onDone, onError }) {
  const res = await fetch(`${API_BASE}/research/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic, sources: ["web"], thread_id: threadId }),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (value) buffer += decoder.decode(value, { stream: true });

    const chunks = buffer.split(/\r?\n\r?\n/);
    buffer = done ? "" : chunks.pop();

    for (const chunk of chunks) {
      const eventMatch = chunk.match(/event: (.+)/);
      const dataMatch = chunk.match(/data: (.+)/);
      if (!dataMatch) continue;

      const event = eventMatch?.[1]?.trim() ?? "message";
      const data = JSON.parse(dataMatch[1]);

      if (event === "node_update") onNode(data);
      else if (event === "done") onDone(data);
      else if (event === "error") onError(data);
    }

    if (done) break;
  }
}