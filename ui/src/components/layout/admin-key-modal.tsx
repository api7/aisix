import { useState } from 'react';

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
          <DialogTitle>Enter Admin API Key</DialogTitle>
          <DialogDescription>
            Your key is stored locally in your browser and never sent to any
            server except the gateway admin API.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-2 py-2">
          <Label htmlFor="admin-key">API Key</Label>
          <Input
            autoFocus
            id="admin-key"
            type="password"
            placeholder="Please enter the admin key"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSave()}
          />
        </div>

        <DialogFooter>
          {current && (
            <Button variant="ghost" onClick={closeModal}>
              Cancel
            </Button>
          )}
          <Button onClick={handleSave} disabled={!draft.trim()}>
            Save Key
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
