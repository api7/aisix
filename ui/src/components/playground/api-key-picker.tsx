import { ChevronDown, KeyRound } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { cn } from '@/lib/utils';

import type { ApiKeyItem } from './types';
import {
  pickerItemBaseClassName,
  pickerMenuBaseClassName,
  pickerTriggerBaseClassName,
} from './picker-styles';

interface ApiKeyPickerProps {
  apiKeys: ApiKeyItem[];
  value: string;
  onChange: (key: string) => void;
}

function maskApiKey(value: string): string {
  if (value.length <= 8) return value;
  return `${value.slice(0, 4)}...${value.slice(-4)}`;
}

export function ApiKeyPicker({ apiKeys, value, onChange }: ApiKeyPickerProps) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    function onDown(e: MouseEvent) {
      if (!ref.current?.contains(e.target as Node)) {
        setOpen(false);
      }
    }

    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  const selected = apiKeys.find((item) => item.key === value);
  const label = selected
    ? maskApiKey(selected.value.key)
    : t('playground.selectApiKey');

  return (
    <div className="relative" ref={ref}>
      <button
        className={cn(pickerTriggerBaseClassName, 'w-48 justify-between')}
        onClick={() => setOpen((prev) => !prev)}
        type="button"
      >
        <KeyRound className="h-4 w-4 text-muted-foreground" />
        <span className="min-w-0 flex-1 truncate text-left">{label}</span>
        <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
      </button>

      {open && (
        <div className={cn(pickerMenuBaseClassName, 'w-72')}>
          {apiKeys.length === 0 ? (
            <div className="px-3 py-2 text-sm text-muted-foreground">
              {t('playground.noApiKeys')}
            </div>
          ) : (
            apiKeys.map((item) => (
              <button
                key={item.key}
                className={cn(
                  pickerItemBaseClassName,
                  item.key === value && 'bg-accent',
                )}
                onClick={() => {
                  onChange(item.key);
                  setOpen(false);
                }}
                type="button"
              >
                <span className="w-full truncate text-left font-medium">
                  {item.key}
                </span>
                <span className="font-mono text-xs text-muted-foreground">
                  {maskApiKey(item.value.key)}
                </span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
