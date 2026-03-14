import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useAdminKey } from '@/hooks/use-admin-key';

export function AdminKeyModal() {
  const { t } = useTranslation();
  const { isModalOpen, closeModal, setKey, key: current } = useAdminKey();
  const [draft, setDraft] = useState(current ?? '');

  function handleSave() {
    const trimmed = draft.trim();
    if (!trimmed) return;
    setKey(trimmed);
    closeModal();
  }

  return (
    <Dialog
      open={isModalOpen}
      onOpenChange={(open) => !open && current && closeModal()}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{t('adminKeyModal.title')}</DialogTitle>
          <DialogDescription>
            {t('adminKeyModal.description')}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-2 py-2">
          <Label htmlFor="admin-key">{t('adminKeyModal.label')}</Label>
          <Input
            autoFocus
            id="admin-key"
            type="password"
            placeholder={t('adminKeyModal.placeholder')}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSave()}
          />
        </div>

        <DialogFooter>
          {current && (
            <Button variant="ghost" onClick={closeModal}>
              {t('adminKeyModal.cancel')}
            </Button>
          )}
          <Button onClick={handleSave} disabled={!draft.trim()}>
            {t('adminKeyModal.save')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
