import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

import en from './locales/en.json';
import zhCN from './locales/zh-CN.json';

const LANG_STORAGE_KEY = 'aisix-ui-language';

function normalizeLanguage(lang: string | null | undefined) {
  if (!lang) return null;
  const lower = lang.toLowerCase();
  if (lower.startsWith('zh')) return 'zh-CN';
  if (lower.startsWith('en')) return 'en';
  return null;
}

function getInitialLanguage() {
  if (typeof window === 'undefined') return 'en';

  const stored = window.localStorage.getItem(LANG_STORAGE_KEY);

  // Manual override has higher priority.
  const normalizedStored = normalizeLanguage(stored);
  if (normalizedStored) return normalizedStored;

  // Follow browser language when there is no local override.
  const browserLang =
    normalizeLanguage(window.navigator.language) ??
    window.navigator.languages.map(normalizeLanguage).find(Boolean);

  if (browserLang) return browserLang;
  return 'en';
}

i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    'zh-CN': { translation: zhCN },
  },
  lng: getInitialLanguage(),
  fallbackLng: 'en',
  interpolation: {
    escapeValue: false, // React already escapes values
  },
});

export default i18n;
