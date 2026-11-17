import assert from 'node:assert/strict'
import { Buffer } from 'node:buffer'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import ts from 'typescript'

// Use the project's compiler so tests do not require native TypeScript support
// in Node. The actual browser module runs with its default empty API base URL.
const source = await readFile(new URL('../src/services/api.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
})
const { imageUrl } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled.outputText).toString('base64')}`
)

for (const [input, expected] of [
  ['page_1.png', '/images/page_1.png'],
  ['ingest-abc/page_1.png', '/images/ingest-abc/page_1.png'],
  ['ingest-abc/page #1.png', '/images/ingest-abc/page%20%231.png'],
  ['ingest-abc\\page_1.png', '/images/ingest-abc/page_1.png'],
  ['C:\\legacy\\images\\page_1.png', '/images/page_1.png'],
  ['/legacy/images/page_1.png', '/images/page_1.png'],
]) {
  test(`imageUrl preserves the public image reference for ${input}`, () => {
    assert.equal(imageUrl(input), expected)
  })
}
