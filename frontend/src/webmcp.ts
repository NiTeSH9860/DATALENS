export type DatasetSnapshot = {
  filename: string
  profile: Record<string, unknown>
  column_types: Record<string, string>
  preview: Record<string, unknown>[]
}

type ModelContext = {
  registerTool: (tool: {
    name: string
    title: string
    description: string
    inputSchema: Record<string, unknown>
    execute: (input: Record<string, unknown>) => Promise<unknown>
    annotations?: { readOnlyHint?: boolean; untrustedContentHint?: boolean }
  }) => Promise<void>
}

declare global {
  interface Document {
    modelContext?: ModelContext
  }
}

let toolsRegistered = false

export function registerWebMcpTools(
  getSnapshot: () => DatasetSnapshot | null,
  askQuestion: (question: string) => Promise<unknown>,
): boolean {
  const modelContext = document.modelContext
  if (!modelContext) return false
  if (toolsRegistered) return true
  toolsRegistered = true

  void modelContext.registerTool({
    name: 'get_dataset_summary',
    title: 'Get dataset summary',
    description: 'Returns the profile and a small preview of the currently loaded CSV dataset.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true, untrustedContentHint: true },
    execute: async () => getSnapshot() ?? { error: 'No dataset is loaded.' },
  })

  void modelContext.registerTool({
    name: 'ask_dataset_question',
    title: 'Ask the dataset a question',
    description: 'Runs a natural-language question through DataLens validated dataset operations.',
    inputSchema: {
      type: 'object',
      properties: { question: { type: 'string', minLength: 1, maxLength: 500 } },
      required: ['question'],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true, untrustedContentHint: true },
    execute: async (input) => {
      const question = typeof input.question === 'string' ? input.question.trim() : ''
      if (!question || question.length > 500) return { error: 'Question must be 1-500 characters.' }
      return askQuestion(question)
    },
  })

  return true
}
