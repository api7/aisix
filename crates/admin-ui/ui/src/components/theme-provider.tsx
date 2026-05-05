/* eslint-disable react-refresh/only-export-components */
import * as React from 'react';
import { ThemeProvider as NextThemesProvider, useTheme } from 'next-themes';

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      storageKey="aisix-ui-theme"
      disableTransitionOnChange
    >
      {children}
    </NextThemesProvider>
  );
}

export { useTheme };
