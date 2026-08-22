import { api } from "./api.js";

export class ChatPanel {
  constructor({ logEl, formEl, inputEl, getSessionId, getUserId }) {
    this.logEl = logEl;
    this.formEl = formEl;
    this.inputEl = inputEl;
    this.getSessionId = getSessionId;
    this.getUserId = getUserId;

    // Disabled by default -- main.js calls setEnabled(true) once a
    // backend session actually exists. Previously this class silently did
    // nothing on send() if there was no session yet, which looked exactly
    // like "the button is broken" with zero feedback. Now it's visibly
    // disabled instead, and re-enabling happens the moment it's usable.
    this.setEnabled(false);

    this.formEl.addEventListener("submit", (e) => {
      e.preventDefault();
      this.send();
    });
  }

  setEnabled(enabled) {
    this.inputEl.disabled = !enabled;
    const btn = this.formEl.querySelector("button");
    if (btn) btn.disabled = !enabled;
    this.inputEl.placeholder = enabled
      ? "Ask the coach about your form..."
      : "Enable your camera first to start chatting...";
  }

  addMessage(role, text, citations = []) {
    const div = document.createElement("div");
    div.className = `chat-msg ${role}`;

    const textSpan = document.createElement("div");
    textSpan.className = "msg-text";
    // Replace **bold** with <strong> and preserve line breaks
    const formatted = (text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n\n+/g, "<br/><br/>")
      .replace(/\n/g, "<br/>");
    textSpan.innerHTML = formatted;
    div.appendChild(textSpan);

    if (citations && citations.length > 0) {
      // Filter out duplicate or empty citation titles
      const uniqueCites = [...new Set(citations.filter(Boolean))];
      if (uniqueCites.length > 0) {
        const cites = document.createElement("div");
        cites.className = "cites";
        cites.innerHTML = `<span class="cite-icon">📚</span> ${uniqueCites.join(" · ")}`;
        div.appendChild(cites);
      }
    }
    this.logEl.appendChild(div);
    this.logEl.scrollTop = this.logEl.scrollHeight;
  }

  async send() {
    const message = this.inputEl.value.trim();
    if (!message) return;

    const sessionId = this.getSessionId();
    if (!sessionId) {
      // Should be unreachable now that the input is disabled without a
      // session, but keep a visible fallback instead of a silent return
      // in case setEnabled() ever gets out of sync with reality.
      this.addMessage("assistant", "No active session yet -- enable your camera first, then try again.");
      return;
    }

    this.addMessage("user", message);
    this.inputEl.value = "";

    try {
      const resp = await api.chat(sessionId, this.getUserId(), message);
      this.addMessage("assistant", resp.reply, resp.citations);
    } catch (err) {
      this.addMessage("assistant", `(Couldn't reach the coach backend: ${err.message}. Is uvicorn running on port 8000?)`);
    }
  }
}
