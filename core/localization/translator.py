"""
Translation/Localization system for Puppets.
Supports PL, EN, FR languages.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any


class Translator:
    """
    Manages translations for multiple languages.
    Singleton pattern to ensure single instance across application.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.current_language = 'EN'
        self.translations: Dict[str, Dict[str, str]] = {}
        self.available_languages = ['EN', 'PL', 'FR']

        # Load all translations
        self.load_translations()

    def load_translations(self):
        """Load translation files for all languages."""
        translations_dir = Path(__file__).parent / 'translations'

        for lang in self.available_languages:
            lang_file = translations_dir / f'{lang.lower()}.json'
            if lang_file.exists():
                with open(lang_file, 'r', encoding='utf-8') as f:
                    self.translations[lang] = json.load(f)
            else:
                print(f"Warning: Translation file not found: {lang_file}")
                self.translations[lang] = {}

    def set_language(self, language: str):
        """
        Set current language.

        Args:
            language: Language code (EN, PL, FR)
        """
        if language in self.available_languages:
            self.current_language = language
        else:
            print(f"Warning: Language '{language}' not supported. Using EN.")
            self.current_language = 'EN'

    def get_language(self) -> str:
        """Get current language code."""
        return self.current_language

    def tr(self, key: str, **kwargs) -> str:
        """
        Translate a key to current language.

        Args:
            key: Translation key (e.g., 'menu.file', 'button.start')
            **kwargs: Format arguments for string formatting

        Returns:
            Translated string
        """
        # Get translation from current language
        translations = self.translations.get(self.current_language, {})

        # Navigate nested keys (e.g., 'menu.file' -> translations['menu']['file'])
        keys = key.split('.')
        value = translations

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                break

        # If translation not found, return key itself
        if value is None or not isinstance(value, str):
            print(f"Warning: Translation missing for key '{key}' in {self.current_language}")
            return key

        # Format string with kwargs if provided
        if kwargs:
            try:
                return value.format(**kwargs)
            except KeyError as e:
                print(f"Warning: Missing format argument {e} for key '{key}'")
                return value

        return value


# Global translator instance
_translator = Translator()


def tr(key: str, **kwargs) -> str:
    """
    Global translation function (shorthand).

    Args:
        key: Translation key
        **kwargs: Format arguments

    Returns:
        Translated string
    """
    return _translator.tr(key, **kwargs)


def set_language(language: str):
    """Set global language."""
    _translator.set_language(language)


def get_language() -> str:
    """Get current language."""
    return _translator.get_language()


def get_translator() -> Translator:
    """Get global translator instance."""
    return _translator
