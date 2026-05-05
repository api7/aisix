import { Bot, ChevronDown } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { cn } from '@/lib/utils';

import {
  pickerItemBaseClassName,
  pickerMenuBaseClassName,
  pickerTriggerBaseClassName,
} from './picker-styles';

import type { ModelItem } from './types';

interface ModelPickerProps {
  label: string;
  models: ModelItem[];
  value: string;
  buttonClassName?: string;
  onChange: (key: string) => void;
}

export function ModelPicker({
  label,
  models,
  value,
  buttonClassName,
  onChange,
}: ModelPickerProps) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        className={cn(pickerTriggerBaseClassName, buttonClassName)}
        onClick={() => setOpen((o) => !o)}
        type="button"
      >
        <Bot className="h-4 w-4 text-muted-foreground" />
        <span>{label}</span>
        <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
      </button>

      {open && (
        <div className={pickerMenuBaseClassName}>
          {models.length === 0 ? (
            <div className="px-3 py-2 text-sm text-muted-foreground">
              {t('playground.noModels')}
            </div>
          ) : (
            models.map((m) => (
              <button
                key={m.key}
                className={cn(
                  pickerItemBaseClassName,
                  m.key === value && 'bg-accent',
                )}
                type="button"
                onClick={() => {
                  onChange(m.key);
                  setOpen(false);
                }}
              >
                <span className="font-medium">{m.value.name}</span>
                <span className="font-mono text-xs text-muted-foreground">
                  {m.value.model}
                </span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
