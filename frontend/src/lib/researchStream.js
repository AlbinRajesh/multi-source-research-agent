export async function streamResearch(query, { onNode, onDone, onError }) {
  const res = await fetch("http://localhost:8000/research/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const chunks = buffer.split("\n\n");
    buffer = chunks.pop(); // last partial chunk stays in buffer

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
  }
}