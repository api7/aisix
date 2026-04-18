import clsx from "clsx";

interface ErrorBannerProps {
  message: string | null;
  onDismiss?: () => void;
}

// Surfaces an admin API error envelope (`{error_msg}`) as a dismissible
// banner. Returns null when there's nothing to show so callers can
// always mount it unconditionally.
export function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  if (!message) return null;
  return (
    <div
      role="alert"
      className={clsx(
        "rounded-md border px-4 py-3 text-sm",
        "border-red-300 bg-red-50 text-red-900",
        "dark:border-red-700 dark:bg-red-950 dark:text-red-100",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <span className="break-words">{message}</span>
        {onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            className="text-red-700 hover:text-red-900 dark:text-red-300 dark:hover:text-red-100"
            aria-label="Dismiss error"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
}
