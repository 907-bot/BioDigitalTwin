"use client";
import { useEffect, useRef, useState } from "react";
import { api, ChatReply } from "@/lib/api";
import { Card, ErrorBox } from "@/components/Panels";

const SUGGESTIONS = [
  "Cohort overview",
  "Tell me about patient P000001",
  "Simulate T2D for P000001 with metformin",
  "What would happen to P000001 glucose if we set bmi to 22?",
  "Estimate ATE of exercise_30m on glucose",
  "CATE of weight_loss on systolic_bp by age",
];

export default function ChatPage() {
  const [session] = useState("ui-" + Math.random().toString(36).slice(2, 8));
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatReply[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<any>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  async function send(text?: string) {
    const msg = (text ?? input).trim();
    if (!msg) return;
    setInput(""); setBusy(true); setErr(null);
    try {
      const reply = await api.chat(session, msg);
      setMessages(m => [...m, reply]);
    } catch (e) { setErr(e); }
    finally { setBusy(false); }
  }

  async function reset() {
    setBusy(true); setErr(null);
    try { await api.chatReset(session); setMessages([]); }
    catch (e) { setErr(e); }
    finally { setBusy(false); }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-text">LLM agent</h1>
        <div className="flex items-center gap-3 text-xs text-muted">
          <span>session: <code className="text-text">{session}</code></span>
          <button onClick={reset} className="btn">Reset</button>
        </div>
      </div>
      <p className="text-sm text-muted">
        Ollama llama3.1 with 9 tool-calling capabilities. Asks a question, picks a tool,
        executes it, and summarises.
      </p>
      {err && <ErrorBox err={err} />}

      <Card>
        <div ref={scrollRef} className="h-[480px] overflow-y-auto pr-2 space-y-4">
          {messages.length === 0 && (
            <div className="text-muted text-sm">
              Ask anything about the cohort, a specific patient, or a disease trajectory.
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className="space-y-2">
              <div className="flex justify-end">
                <div className="max-w-[80%] bg-teal/15 border border-teal/30 rounded-lg px-3 py-2 text-sm">
                  {m.user_message}
                </div>
              </div>
              <div className="flex justify-start">
                <div className="max-w-[80%] bg-panel2 border border-border rounded-lg px-3 py-2 text-sm space-y-2">
                  {m.tool_calls.length > 0 && (
                    <details open className="text-xs">
                      <summary className="cursor-pointer text-purple">
                        🔧 {m.tool_calls.length} tool call{m.tool_calls.length > 1 ? "s" : ""}
                      </summary>
                      <div className="mt-1 space-y-1 font-mono text-[11px] text-muted">
                        {m.tool_calls.map((t, j) => (
                          <div key={j}>
                            <span className="text-teal">{t.tool}</span>({JSON.stringify(t.args)})
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                  <div className="whitespace-pre-wrap text-text">{m.reply}</div>
                  <div className="text-[10px] text-muted">
                    {m.backend} · {m.elapsed_s}s
                  </div>
                </div>
              </div>
            </div>
          ))}
          {busy && <div className="text-xs text-muted">thinking…</div>}
        </div>
      </Card>

      <Card>
        <div className="flex gap-2">
          <input className="input flex-1" placeholder="Ask the agent…"
                 value={input}
                 onChange={e => setInput(e.target.value)}
                 onKeyDown={e => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), send())}
                 disabled={busy} />
          <button onClick={() => send()} disabled={busy || !input.trim()} className="btn-primary">
            Send
          </button>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {SUGGESTIONS.map(s => (
            <button key={s} onClick={() => send(s)} disabled={busy}
                    className="text-xs px-2 py-1 rounded border border-border
                               bg-panel2 text-muted hover:text-text hover:border-teal/30">
              {s}
            </button>
          ))}
        </div>
      </Card>
    </div>
  );
}
