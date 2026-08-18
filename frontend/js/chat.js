import { api } from "./api.js";

export class ChatPanel {
  constructor({ logEl, formEl, inputEl, getSessionId, getUserId }) {
    this.logEl = logEl;
    this.formEl = formEl;
    this.inputEl = inputEl;
    this.getSessionId = getSessionId;
    this.getUserId = getUserId;

    this.formEl.addEventListener("submit", (e) => {
      e.preventDefault();
      this.send();
    });
  }

  addMessage(role, text, citations = []) {
    const div = document.createElement("div");
    div.className = `chat-msg ${role}`;
    div.textContent = text;
    if (citations.length) {
      const cites = document.createElement("div");
      cites.className = "cites";
      cites.textContent = `Sources: ${citations.join(", ")}`;
      div.appendChild(cites);
    }
    this.logEl.appendChild(div);
    this.logEl.scrollTop = this.logEl.scrollHeight;
  }

  async send() {
    const message = this.inputEl.value.trim();
    const sessionId = this.getSessionId();
    if (!message || !sessionId) return;

    this.addMessage("user", message);
    this.inputEl.value = "";

    try {
      const resp = await api.chat(sessionId, this.getUserId(), message);
      this.addMessage("assistant", resp.reply, resp.citations);
    } catch (err) {
      this.addMessage("assistant", `(Couldn't reach the coach backend: ${err.message})`);
    }
  }
}
