import Editor, { loader } from '@monaco-editor/react';
import editorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
import * as monaco from 'monaco-editor';
import jsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker';
import 'monaco-editor/esm/vs/language/json/monaco.contribution.js';

import { useTheme } from '@/components/theme-provider';
import { cn } from '@/lib/utils';

type MonacoEnv = {
  MonacoEnvironment?: {
    getWorker: (_moduleId: string, label: string) => Worker;
  };
};

const env = globalThis as MonacoEnv;
if (!env.MonacoEnvironment) {
  env.MonacoEnvironment = {
    getWorker(_moduleId: string, label: string) {
      if (label === 'json') {
        return new jsonWorker();
      }
      return new editorWorker();
    },
  };
}

loader.config({ monaco });

interface MonacoJsonEditorProps {
  value: string;
  onChange: (value: string) => void;
  className?: string;
  height?: number | string;
  ariaLabel?: string;
}

export function MonacoJsonEditor({
  value,
  onChange,
  className,
  height = 240,
  ariaLabel = 'JSON editor',
}: MonacoJsonEditorProps) {
  const { resolvedTheme } = useTheme();
  const monacoTheme = resolvedTheme === 'dark' ? 'vs-dark' : 'vs';

  return (
    <div
      className={cn(
        'overflow-hidden rounded-md border bg-background',
        className,
      )}
    >
      <Editor
        language="json"
        theme={monacoTheme}
        value={value}
        height={height}
        loading={null}
        options={{
          automaticLayout: true,
          ariaLabel,
          minimap: { enabled: false },
          wordWrap: 'on',
          lineNumbersMinChars: 3,
          padding: { top: 10, bottom: 10 },
          scrollBeyondLastLine: false,
          tabSize: 2,
          insertSpaces: true,
          detectIndentation: false,
          overviewRulerBorder: false,
          renderValidationDecorations: 'on',
          fontSize: 13,
        }}
        beforeMount={(m) => {
          // json language may not be ready yet during SSR/fast mounts
          const jsonLang = m.languages.json as
            | typeof m.languages.json
            | undefined;
          jsonLang?.jsonDefaults?.setDiagnosticsOptions({
            validate: true,
            allowComments: false,
            trailingCommas: 'error',
            schemaValidation: 'ignore',
            enableSchemaRequest: false,
          });
          jsonLang?.jsonDefaults?.setModeConfiguration({
            documentFormattingEdits: true,
            documentRangeFormattingEdits: true,
            completionItems: true,
            hovers: true,
            diagnostics: true,
            tokens: true,
            colors: false,
            foldingRanges: true,
            selectionRanges: true,
          });
        }}
        onChange={(next) => onChange(next ?? '')}
      />
    </div>
  );
}
