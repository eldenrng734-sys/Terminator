const $ = (id) => document.getElementById(id);
const stage = $("stage");
const thread = $("thread");
const input = $("input");
const imageInput = $("image-input");
const attachments = $("attachments");
const status = $("status");
const side = $("side");
const signInModal = $("signin-modal");
const signInDismissKey = "signin_prompt_dismissed";

let activeSessionId = null;
let sessions = [];
let pendingImages = [];
let busy = false;
let searchGeneration = 0;
const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = SpeechRecognitionAPI ? new SpeechRecognitionAPI() : null;
let listening = false;
let activeSpeechButton = null;
let isAuthenticated = false;

class AuthenticationRequiredError extends Error {}

function showLoggedOutState() {
  isAuthenticated = false;
  $("profile-user").classList.add("hide");
  $("profile-login").classList.remove("hide");
}

function wasSignInPromptDismissed() {
  try {
    return sessionStorage.getItem(signInDismissKey) === "true";
  } catch {
    return false;
  }
}

function scheduleSignInPrompt() {
  if (wasSignInPromptDismissed()) return;
  window.setTimeout(() => {
    if (!isAuthenticated && !wasSignInPromptDismissed()) {
      signInModal.showModal();
      $("signin-close").focus();
    }
  }, 5000);
}

function dismissSignInPrompt() {
  try {
    sessionStorage.setItem(signInDismissKey, "true");
  } catch {
    // Keep the prompt dismissible if session storage is unavailable.
  }
  if (signInModal.open) signInModal.close();
}

async function apiFetch(resource, options) {
  const response = await fetch(resource, options);
  if (response.status === 401) {
    showLoggedOutState();
    throw new AuthenticationRequiredError("Authentication required.");
  }
  return response;
}

function reportError(error) {
  if (!(error instanceof AuthenticationRequiredError)) {
    setStatus(error.message);
  }
}

if (recognition) {
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  recognition.onstart = () => {
    setMicState(true);
    setStatus("Listening...");
  };
  recognition.onresult = (event) => {
    const transcript = event.results?.[0]?.[0]?.transcript;
    if (transcript) void sendMessage(transcript);
  };
  recognition.onerror = (event) => {
    const messages = {
      "not-allowed": "Microphone access denied.",
      "no-speech": "No speech detected. Try again.",
    };
    setStatus(messages[event.error] || `Speech recognition error: ${event.error}`);
  };
  recognition.onend = () => {
    setMicState(false);
    if (status.textContent === "Listening...") setStatus("");
  };
} else {
  $("mic").disabled = true;
  $("mic").title = "Speech recognition isn't supported in this browser";
  $("mic").setAttribute("aria-label", "Speech recognition isn't supported in this browser");
}

async function readJson(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || data.error || "Request failed.");
  return data;
}

function setStatus(message = "") {
  status.textContent = message;
}

function scrollToLatest() {
  stage.scrollTop = stage.scrollHeight;
}

function messageParts(content) {
  if (typeof content === "string") return { text: content, images: [] };
  const parts = Array.isArray(content) ? content : [];
  return {
    text: parts.filter((part) => part.type === "text").map((part) => part.text).join("\n"),
    images: parts
      .filter((part) => part.type === "image_url")
      .map((part) => part.image_url?.url)
      .filter((url) => typeof url === "string" && (/^data:image\/(png|jpeg|webp);base64,/.test(url) || url.startsWith("blob:"))),
  };
}

function renderBotMessage(messageElement, botResponse) {
  const response = String(botResponse ?? "");
  try {
    if (window.marked?.parse && window.DOMPurify?.sanitize) {
      messageElement.innerHTML = window.DOMPurify.sanitize(window.marked.parse(response));
      messageElement.classList.add("bot-message");
      return;
    }
  } catch {
    // Fall through to safe plain-text display if a renderer fails.
  }
  messageElement.textContent = response;
}

function addMessage(message, { typing = false } = {}) {
  const element = document.createElement("article");
  const role = message.role === "user" ? "user" : "bot";
  element.className = `msg ${role}`;
  if (typing) {
    element.setAttribute("aria-label", "Assistant is responding");
    element.innerHTML = '<span class="dots" aria-hidden="true"><span></span><span></span><span></span></span>';
  } else {
    const { text, images } = messageParts(message.content);
    if (text) {
      const content = document.createElement("div");
      content.className = "msg-content";
      if (role === "bot") renderBotMessage(content, text);
      else content.textContent = text;
      element.append(content);
    }
    images.forEach((src) => {
      const image = document.createElement("img");
      image.className = "msg-image";
      image.src = src;
      image.alt = "Attached image";
      if (src.startsWith("blob:")) {
        const releaseUrl = () => URL.revokeObjectURL(src);
        image.onload = releaseUrl;
        image.onerror = releaseUrl;
      }
      element.append(image);
    });
    if (role === "bot" && text) {
      const speak = document.createElement("button");
      speak.className = "speak";
      speak.type = "button";
      speak.setAttribute("aria-label", "Read response aloud");
      speak.title = "Read response aloud";
      speak.innerHTML = '<svg class="i" aria-hidden="true"><use href="#icon-volume"></use></svg>';
      speak.onclick = () => playVoiceReply(text, speak);
      element.append(speak);
    }
  }
  thread.append(element);
  scrollToLatest();
  return element;
}

function renderMessages(messages) {
  thread.replaceChildren();
  messages.forEach((message) => addMessage(message));
  const empty = messages.length === 0;
  $("welcome").classList.toggle("hide", !empty);
  thread.classList.toggle("hide", empty);
  $("chips").classList.toggle("hide", !empty);
  if (!empty) scrollToLatest();
}

function renderRecent(items) {
  const recent = $("recent");
  recent.replaceChildren();
  if (items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-recent";
    empty.textContent = $("find").value ? "No matching conversations" : "No conversations yet";
    recent.append(empty);
    return;
  }
  items.forEach((session) => {
    const row = document.createElement("div");
    row.className = "recent-item";
    row.setAttribute("role", "listitem");
    const button = document.createElement("button");
    button.className = "recent-open";
    button.type = "button";
    button.setAttribute("aria-current", String(session.id === activeSessionId));
    button.textContent = session.title || "New conversation";
    button.title = session.title || "New conversation";
    button.onclick = () => openSession(session.id);
    const remove = document.createElement("button");
    remove.className = "recent-delete";
    remove.type = "button";
    remove.disabled = busy;
    remove.setAttribute("aria-label", `Delete conversation: ${session.title || "New conversation"}`);
    remove.title = "Delete conversation";
    remove.innerHTML = '<svg class="i" aria-hidden="true"><use href="#icon-close"></use></svg>';
    remove.onclick = () => void deleteSession(session.id);
    row.append(button, remove);
    recent.append(row);
  });
}

async function deleteSession(sessionId) {
  const session = sessions.find((item) => item.id === sessionId);
  const title = session?.title || "this conversation";
  if (!window.confirm(`Delete "${title}"? This cannot be undone.`)) return;

  try {
    await readJson(await apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" }));
    const wasActive = activeSessionId === sessionId;
    if (wasActive) {
      activeSessionId = null;
      localStorage.removeItem("active_session_id");
      renderMessages([]);
    }
    await refreshSessions($("find").value);
    if (wasActive && sessions.length) await openSession(sessions[0].id);
    setStatus("");
  } catch (error) {
    reportError(error);
  }
}

async function refreshSessions(query = "") {
  const generation = ++searchGeneration;
  try {
    const response = await apiFetch(`/api/sessions?q=${encodeURIComponent(query)}`);
    const items = await readJson(response);
    if (generation !== searchGeneration) return;
    sessions = items;
    renderRecent(items);
  } catch (error) {
    reportError(error);
  }
}

async function openSession(sessionId) {
  try {
    const response = await apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
    const session = await readJson(response);
    activeSessionId = session.id;
    localStorage.setItem("active_session_id", activeSessionId);
    renderMessages(session.messages || []);
    renderRecent(sessions);
    setStatus("");
    if (window.matchMedia("(max-width: 760px)").matches) side.classList.add("closed");
  } catch (error) {
    reportError(error);
  }
}

async function createSession() {
  const response = await apiFetch("/api/sessions", { method: "POST" });
  const session = await readJson(response);
  activeSessionId = session.id;
  localStorage.setItem("active_session_id", activeSessionId);
  renderMessages([]);
  await refreshSessions($("find").value);
  input.focus();
  if (window.matchMedia("(max-width: 760px)").matches) side.classList.add("closed");
}

function updateComposer() {
  $("send").classList.toggle("on", Boolean(input.value.trim() || pendingImages.length));
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

function renderAttachments() {
  attachments.replaceChildren();
  attachments.classList.toggle("hide", pendingImages.length === 0);
  pendingImages.forEach((file, index) => {
    const item = document.createElement("span");
    item.className = "attachment";
    const name = document.createElement("span");
    name.className = "attachment-name";
    name.textContent = file.name;
    const remove = document.createElement("button");
    remove.className = "attachment-remove";
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${file.name}`);
    remove.onclick = () => {
      pendingImages.splice(index, 1);
      renderAttachments();
      updateComposer();
    };
    item.append(name, remove);
    attachments.append(item);
  });
}

// Keep this adapter as the single UI-to-chat integration point when replacing the chat API.
async function requestAssistantReply(formData) {
  return readJson(await apiFetch("/api/chat", { method: "POST", body: formData }));
}

async function sendMessage(text) {
  const messageText = text.trim();
  if ((!messageText && pendingImages.length === 0) || busy) return;
  if (!activeSessionId) {
    try {
      await createSession();
    } catch (error) {
      reportError(error);
      return;
    }
  }

  const sentImages = [...pendingImages];
  const userContent = sentImages.length
    ? [
        { type: "text", text: messageText },
        ...sentImages.map((file) => ({ type: "image_url", image_url: { url: URL.createObjectURL(file) } })),
      ]
    : messageText;
  $("welcome").classList.add("hide");
  $("chips").classList.add("hide");
  thread.classList.remove("hide");
  addMessage({ role: "user", content: userContent });
  input.value = "";
  pendingImages = [];
  imageInput.value = "";
  renderAttachments();
  updateComposer();
  busy = true;
  $("send").disabled = true;
  setStatus("Thinking...");
  const typing = addMessage({ role: "assistant", content: "" }, { typing: true });
  const formData = new FormData();
  formData.append("message", messageText);
  formData.append("session_id", activeSessionId);
  sentImages.forEach((file) => formData.append("images", file));
  const requestSessionId = activeSessionId;

  try {
    const result = await requestAssistantReply(formData);
    typing.remove();
    if (activeSessionId === requestSessionId) {
      addMessage({ role: "assistant", content: result.reply });
    }
    await refreshSessions($("find").value);
    if (activeSessionId === requestSessionId) setStatus("");
  } catch (error) {
    typing.remove();
    if (!(error instanceof AuthenticationRequiredError) && activeSessionId === requestSessionId) {
      addMessage({ role: "assistant", content: `I couldn't send that message: ${error.message}` });
      setStatus(error.message);
    }
  } finally {
    busy = false;
    $("send").disabled = false;
  }
}

function setMicState(recording) {
  listening = recording;
  const mic = $("mic");
  mic.classList.toggle("rec", recording);
  mic.setAttribute("aria-pressed", String(recording));
  mic.setAttribute("aria-label", recording ? "Stop voice input" : "Start voice input");
  mic.title = recording ? "Stop voice input" : "Start voice input";
}

function resetSpeechButton(button) {
  button.disabled = false;
  button.title = "Read response aloud";
  button.setAttribute("aria-label", "Read response aloud");
}

function playVoiceReply(text, button) {
  if (!("speechSynthesis" in window) || typeof window.SpeechSynthesisUtterance !== "function") {
    setStatus("Voice playback isn't supported in this browser.");
    return;
  }

  if (activeSpeechButton) resetSpeechButton(activeSpeechButton);
  window.speechSynthesis.cancel();
  const utterance = new window.SpeechSynthesisUtterance(text);
  activeSpeechButton = button;
  button.disabled = true;
  button.title = "Speaking response";
  button.setAttribute("aria-label", "Speaking response");
  const finish = () => {
    resetSpeechButton(button);
    if (activeSpeechButton === button) activeSpeechButton = null;
  };
  utterance.onend = finish;
  utterance.onerror = () => {
    finish();
    setStatus("Voice playback failed.");
  };
  try {
    window.speechSynthesis.speak(utterance);
  } catch {
    finish();
    setStatus("Voice playback failed.");
  }
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  void sendMessage(input.value);
});
input.addEventListener("input", updateComposer);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    void sendMessage(input.value);
  }
});
imageInput.addEventListener("change", () => {
  pendingImages = [...pendingImages, ...Array.from(imageInput.files || [])];
  renderAttachments();
  updateComposer();
});
$("attach").onclick = () => imageInput.click();
$("images").onclick = () => {
  if (!input.value.trim()) input.value = "Describe this image.";
  updateComposer();
  imageInput.click();
};
document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.onclick = () => void sendMessage(button.dataset.prompt);
});
$("new").onclick = () => void createSession().catch(reportError);

$("mic").onclick = () => {
  if (!recognition) return;
  try {
    if (listening) recognition.stop();
    else recognition.start();
  } catch {
    setMicState(false);
    setStatus("Speech recognition could not start. Try again.");
  }
};

$("collapse").onclick = () => side.classList.add("closed");
$("menu").onclick = () => side.classList.toggle("closed");
if (window.matchMedia("(max-width: 760px)").matches) side.classList.add("closed");
$("search-btn").onclick = () => {
  const find = $("find");
  find.classList.toggle("hide");
  if (!find.classList.contains("hide")) find.focus();
  else {
    find.value = "";
    void refreshSessions();
  }
};
$("find").addEventListener("input", (event) => void refreshSessions(event.target.value));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && signInModal.open) {
    dismissSignInPrompt();
  }
  if (event.key === "Escape" && window.matchMedia("(max-width: 760px)").matches) side.classList.add("closed");
});

$("signin-close").addEventListener("click", dismissSignInPrompt);
$("signin-later").addEventListener("click", dismissSignInPrompt);
signInModal.addEventListener("click", (event) => {
  if (event.target === signInModal) dismissSignInPrompt();
});
signInModal.addEventListener("cancel", (event) => {
  event.preventDefault();
  dismissSignInPrompt();
});

async function initialize() {
  let user;
  try {
    user = await readJson(await apiFetch("/api/me"));
  } catch (error) {
    if (error instanceof AuthenticationRequiredError) {
      scheduleSignInPrompt();
      return;
    }
    throw error;
  }

  isAuthenticated = true;
  $("profile-login").classList.add("hide");
  $("profile-user").classList.remove("hide");
  $("profile-name").textContent = user.name || user.email || "";
  $("profile-email").textContent = user.email || "";
  await refreshSessions();
  const savedId = localStorage.getItem("active_session_id");
  const saved = sessions.find((session) => session.id === savedId);
  if (saved) await openSession(saved.id);
  else if (sessions.length) await openSession(sessions[0].id);
  else await createSession();
}

void initialize().catch(reportError);