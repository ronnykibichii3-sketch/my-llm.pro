# MyLLM.pro Frontend

## Goal

Build a modern AI Operating System interface.

Not a chatbot.

Not ChatGPT clone.

A complete AI dashboard.

---

# Stack

React 19
TypeScript
Vite
TailwindCSS
Lucide React
Zustand
TanStack Query
React Router
Axios

---

# Design Principles

AI First

Everything revolves around the AI.

---

Modular

Every page must be isolated.

---

Responsive

Desktop first.

Tablet support.

Mobile support later.

---

API First

Frontend never contains business logic.

Frontend only consumes API endpoints.

---

# Layout

┌────────────────────────────────────┐
│ Top Bar                            │
├──────────────┬─────────────────────┤
│ Sidebar      │ Main Workspace      │
│              │                     │
│ Conversations│ Chat Window         │
│ Agents       │                     │
│ Memory       │                     │
│ Automation   │                     │
│ Models       │                     │
│ Settings     │                     │
└──────────────┴─────────────────────┘

---

# Structure

```text
frontend/
│
├── public/
│
├── src/
│
│   ├── assets/
│
│   ├── components/
│
│   ├── layout/
│   │   ├── Sidebar.tsx
│   │   ├── Topbar.tsx
│   │   └── AppLayout.tsx
│
│   ├── pages/
│   │   ├── ChatPage.tsx
│   │   ├── MemoryPage.tsx
│   │   ├── AgentsPage.tsx
│   │   ├── AutomationPage.tsx
│   │   ├── ModelsPage.tsx
│   │   ├── SettingsPage.tsx
│   │   ├── LoginPage.tsx
│   │   └── RegisterPage.tsx
│
│   ├── modules/
│   │
│   │   ├── chat/
│   │   │   ├── components/
│   │   │   ├── hooks/
│   │   │   ├── services/
│   │   │   ├── store/
│   │   │   └── types/
│   │
│   │   ├── conversations/
│   │
│   │   ├── auth/
│   │
│   │   └── system/
│
│   ├── services/
│
│   ├── store/
│
│   ├── hooks/
│
│   ├── routes/
│
│   ├── lib/
│
│   ├── types/
│
│   ├── App.tsx
│   └── main.tsx
│
├── package.json
├── vite.config.ts
├── tsconfig.json
└── tailwind.config.ts
```

---

# Pages

Dashboard

Chat

Memory

Agents

Automation

Models

Settings

Login

Register

---

# State Management

Zustand

Stores:

authStore

chatStore

conversationStore

settingsStore

systemStore

---

# API Layer

services/

auth.service.ts

chat.service.ts

conversation.service.ts

system.service.ts

Never perform fetch calls directly inside components.

---

# Routing

/

Chat

/login

/register

/memory

/agents

/automation

/models

/settings

---

# Error Handling

Global error boundary.

Toast notifications.

Loading indicators.

Retry strategy.

---

# Authentication

JWT

Stored securely.

Automatic refresh handling.

Protected routes.

---

# Theme

Dark Mode

Primary:

#00E5FF

Background:

#0F172A

Surface:

#111827

Accent:

#38BDF8

Success:

#22C55E

Danger:

#EF4444

---

# Chat Features

Conversation list

Message streaming

Markdown support

Code highlighting

Auto scroll

Agent status indicator

Typing indicator

Conversation search

Future voice support

---

# Future Reserved Panels

Memory Panel

Training Panel

Agent Monitor

System Metrics

Prompt Analytics