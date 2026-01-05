//! # Yoruba Phonology & Syllabizer
//!
//! A strictly typed, linguistically accurate representation of the Yoruba language sound system.
//!
//! This library provides primitives to parse, analyze, and reconstruct Yoruba words based on
//! standard phonological rules.
//!
//! ## 📚 Linguistic Overview
//!
//! ### 1. Vowels (*Fáwé̩lì*)
//! Yoruba has a total of seven oral vowels and five nasal vowels.
//! - **Oral**: `a`, `e`, `ẹ`, `i`, `o`, `ọ`, `u`.
//! - **Nasal**: `an`, `ẹn`, `in`, `ọn`, `un`. (Marked with an accompanying `n`).
//!
//! ### 2. Tones (*Ohùn*)
//! Yoruba is a tonal language. The meaning of a word changes based on the pitch.
//! - **Low (`Do`)**: Marked with a grave accent (e.g., `à`).
//! - **Mid (`Re`)**: Usually unmarked (e.g., `a`), but strictly modeled here.
//! - **High (`Mi`)**: Marked with an acute accent (e.g., `á`).
//!
//! ### 3. Syllable Structure
//! Yoruba syllables are open (they end in a vowel or a nasal nucleus).
//! - **V**: Vowel only (e.g., `ẹ` in *ẹ-nu* (mouth)).
//! - **CV**: Consonant + Vowel (e.g., `ji` in *ji-n* (deep)).
//! - **N**: Syllabic Nasal (e.g., `n` in *n-ǹ-kan* (thing)).
//!
//! ---
//!
//! ## ⚠️ Common Gotchas & Pitfalls
//!
//! ### The "Ambiguous N"
//! The letter `n` is the most complex character in Yoruba parsing. It can be:
//! 1.  **A Consonant**: Starts a syllable (e.g., `n` in *ni*).
//! 2.  **A Nasal Marker**: Ends a vowel to make it nasal (e.g., `n` in *yin*).
//! 3.  **A Syllabic Nasal**: Stands alone as a syllable (e.g., `n` in *n-kan*).
//!
//! *How we handle it:* The parser uses a lookahead strategy. If `n` is followed by a vowel,
//! it's a consonant. If it's followed by a consonant, it's a syllabic nasal.
//!
//! ### Unicode Normalization
//! Input text varies wildly. `ẹ` might be a single character (`\u{1EB9}`) or two
//! (`e` + `dot_below`).
//!
//! *How we handle it:* The parser automatically normalizes all input to NFD (Decomposed)
//! before processing. You don't need to manually normalize strings before passing them in.
//!
//! ### Mid Tone Marking
//! In standard orthography, the Mid tone is left blank. However, strictly speaking,
//! the tone exists.
//!
//! *How we handle it:* The `Tone::Mid` variant exists explicitly. When parsing `a`,
//! we parse it as `OralVowel::Ah` + `Tone::Mid`.

#![no_std]

#[cfg(any(test, feature = "std"))]
extern crate std;

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod parse;
// TODO:
pub mod sound_parse;

// Re-export core types for ease of use
pub use parse::YorubaChar as Char;
pub use parse::YorubaChars as Chars;
pub use parse::parse_syllables;
pub use parse::parse_words;

use core::fmt;
use parse::{COM_ACUTE, COM_GRAVE, compose_yoruba_lower};

// ============================================================================
// CORE PHONOLOGICAL TYPES
// ============================================================================

/// The three phonemic tones in Yoruba.
///
/// Tones are phonemic because they bring about a change in meaning.
///
/// # Examples
/// ```rust
/// use yoruba::Tone;
///
/// assert_eq!(Tone::Low.mark(), Some('\u{0300}')); // Grave (`).
/// assert_eq!(Tone::High.mark(), Some('\u{0301}')); // Acute (´).
/// assert_eq!(Tone::Mid.mark(), None); // Usually unmarked.
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Tone {
    /// The Mid tone (Normal voice pitch).
    /// Solfège: **Re**.
    /// Representation: Left blank (e.g., `a`).
    #[default]
    Mid,

    /// The High tone.
    /// Solfège: **Mi**.
    /// Representation: Acute accent (´) (e.g., `á`).
    High,

    /// The Low tone.
    /// Solfège: **Do**.
    /// Representation: Grave accent (`) (e.g., `à`).
    Low,
}

impl Tone {
    /// Returns the combining diacritic character for this tone.
    pub const fn mark(&self) -> Option<char> {
        match self {
            Tone::Mid => None,
            Tone::Low => Some(COM_GRAVE),
            Tone::High => Some(COM_ACUTE),
        }
    }
}

/// The nature of the vowel sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Vowel {
    /// Air flows only through the mouth.
    Oral(OralVowel),
    /// Air flows through the nose and mouth (marked with `n`).
    Nasal(NasalVowel),
}

// Boilerplate conversions
impl From<OralVowel> for Vowel {
    fn from(v: OralVowel) -> Self {
        Self::Oral(v)
    }
}
impl From<NasalVowel> for Vowel {
    fn from(v: NasalVowel) -> Self {
        Self::Nasal(v)
    }
}

/// The 7 Oral Vowels of Standard Yoruba.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OralVowel {
    /// **A** (as in *Ajá* - Dog)
    Ah,
    /// **E** (as in *Èdè* - Language)
    Ey,
    /// **Ẹ** (E with dot below) - Open E (as in *Ẹsẹ̀* - Leg).
    /// Phonetic: /ɛ/
    Eh,
    /// **I** (as in *Ilè* - House)
    I,
    /// **O** (as in *Owó* - Money)
    Oh,
    /// **Ọ** (O with dot below) - Open O (as in *Ọmọ* - Child).
    /// Phonetic: /ɔ/
    Or,
    /// **U** (as in *Ìlù* - Drum)
    U,
}

impl OralVowel {
    /// Returns the unicode character for the base vowel (without tone).
    pub fn as_char(&self) -> char {
        match self {
            Self::Ah => 'a',
            Self::Ey => 'e',
            Self::Eh => 'ẹ', // U+1EB9
            Self::I => 'i',
            Self::Oh => 'o',
            Self::Or => 'ọ', // U+1ECC
            Self::U => 'u',
        }
    }
}

/// The 5 Nasal Vowels of Standard Yoruba.
///
/// Note: In Yoruba syllable structure, nasal vowels are not the same as syllabic consonants.
/// e.g. the nasal vowel /in/ in `i-rin` is a vowel, but `n` in `n-lọ` is a syllabic consonant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NasalVowel {
    /// **An** (as in *Ìtàn* - History).
    An,
    /// **Ẹn** (as in *Ìyẹn* - That one).
    En,
    /// **In** (as in *Ìrìn* - Walk/Journey).
    In,
    /// **Ọn** (as in *Ìbọn* - Gun).
    Orn,
    /// **Un** (as in *Fún* - Give).
    Un,

    #[cfg(feature = "dialects")]
    /// **En** - Dialectal variant (often Ekitì/Ìjẹ̀bú).
    Eyn,
}

impl NasalVowel {
    /// Returns the string representation of the nasal vowel.
    pub const fn as_str(&self) -> &'static str {
        match self {
            NasalVowel::An => "an",
            NasalVowel::En => "ẹn",
            NasalVowel::In => "in",
            NasalVowel::Orn => "ọn",
            NasalVowel::Un => "un",
            #[cfg(feature = "dialects")]
            NasalVowel::Eyn => "en",
        }
    }
}

/// The 18 Consonants of Standard Yoruba.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Consonant {
    /// **B** - /b/
    Bi,
    /// **D** - /d/
    Di,
    /// **F** - /f/
    Fi,
    /// **G** - /g/ (Hard G)
    Gi,
    /// **Gb** - /ɡ͡b/ (Labial-velar voiced stop).
    Gbi,
    /// **H** - /h/
    Hi,
    /// **J** - /d͡ʒ/ (English 'J' sound)
    Ji,
    /// **K** - /k/
    Ki,
    /// **L** - /l/
    Li,
    /// **M** - /m/
    Mi,
    /// **N** - /n/
    Ni,
    /// **P** - /k͡p/ (Labial-velar voiceless stop).
    Pi,
    /// **R** - /ɾ/ (Tapped R)
    Ri,
    /// **S** - /s/
    Si,
    /// **Ṣ** (S with dot) - /ʃ/ (English 'sh' sound).
    Shi,
    /// **T** - /t/
    Ti,
    /// **W** - /w/
    Wi,
    /// **Y** - /j/ (English 'y' sound)
    Yi,
}

impl Consonant {
    /// Returns the string representation of the consonant.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Bi => "b",
            Self::Di => "d",
            Self::Fi => "f",
            Self::Gi => "g",
            Self::Gbi => "gb",
            Self::Hi => "h",
            Self::Ji => "j",
            Self::Ki => "k",
            Self::Li => "l",
            Self::Mi => "m",
            Self::Ni => "n",
            Self::Pi => "p",
            Self::Ri => "r",
            Self::Si => "s",
            Self::Shi => "ṣ",
            Self::Ti => "t",
            Self::Wi => "w",
            Self::Yi => "y",
        }
    }
}

/// A Syllabic Consonant.
///
/// Only two consonants, /n/ and /m/, can function as syllables in Yoruba.
/// They carry tone just like vowels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyllabicNasal {
    /// **N** (as in *Aláǹgbá*).
    Hn,
    /// **M** (as in *Bímbọ́lá*).
    Hm,
}

// ============================================================================
// COMPOSITE SOUND UNITS
// ============================================================================

/// A Vowel (Oral or Nasal) bearing a Tone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TonedVowel {
    pub vowel: Vowel,
    pub tone: Tone,
}

impl fmt::Display for TonedVowel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 1. Identify base char and suffix (for nasal 'n')
        let (base, suffix) = match self.vowel {
            Vowel::Oral(o) => (o.as_char(), ""),
            Vowel::Nasal(n) => match n {
                NasalVowel::An => ('a', "n"),
                NasalVowel::En => ('ẹ', "n"),
                NasalVowel::In => ('i', "n"),
                NasalVowel::Orn => ('ọ', "n"),
                NasalVowel::Un => ('u', "n"),
                #[cfg(feature = "dialects")]
                NasalVowel::Eyn => ('e', "n"),
            },
        };

        // 2. Compose base + tone (e.g., 'a' + High -> 'á')
        if let Some(composed) = compose_yoruba_lower(base, self.tone) {
            write!(f, "{}", composed)?;
        } else {
            // Fallback for weird fonts/systems
            write!(f, "{}", base)?;
            if let Some(mark) = self.tone.mark() {
                write!(f, "{}", mark)?;
            }
        }

        // 3. Append nasal marker if exists
        write!(f, "{}", suffix)
    }
}

/// A standard Consonant + Vowel pair (CV).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TonedConsonant(pub Consonant, pub TonedVowel);

/// A Syllabic Nasal bearing a Tone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TonedSyllabicNasal {
    pub syllabic_nasal: SyllabicNasal,
    pub tone: Tone,
}

impl fmt::Display for TonedSyllabicNasal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ch = match self.syllabic_nasal {
            SyllabicNasal::Hn => 'n',
            SyllabicNasal::Hm => 'm',
        };

        if let Some(composed) = compose_yoruba_lower(ch, self.tone) {
            write!(f, "{}", composed)
        } else {
            write!(f, "{}", ch)?;
            if let Some(mark) = self.tone.mark() {
                write!(f, "{}", mark)?;
            }
            Ok(())
        }
    }
}

// ============================================================================
// THE SYLLABLE
// ============================================================================

/// The fundamental rhythmic unit of Yoruba.
///
/// Can be one of:
/// 1.  **Vowel**: (e.g. *a*, *ẹ*)
/// 2.  **Consonant**: (e.g. *ba*, *ji*)
/// 3.  **Syllabic Nasal**: (e.g. *n*, *m*)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Syllable {
    kind: SyllableKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyllableKind {
    /// A standalone vowel syllable. (e.g., the `à` in *àlọ*).
    Vowel(TonedVowel),

    /// A Consonant-Vowel pair. (e.g., `lọ`).
    Consonant(TonedConsonant),

    /// A Syllabic Nasal unit. (e.g., `n` in *n-lò*).
    SyllabicNasal(TonedSyllabicNasal),
}

// Convenient conversion impls
impl From<TonedVowel> for Syllable {
    fn from(v: TonedVowel) -> Self {
        Self::vowel(v)
    }
}
impl From<TonedConsonant> for Syllable {
    fn from(c: TonedConsonant) -> Self {
        Self::consonant(c)
    }
}
impl From<TonedSyllabicNasal> for Syllable {
    fn from(n: TonedSyllabicNasal) -> Self {
        Self::syllabic_nasal(n)
    }
}

impl Syllable {
    /// Creates a Syllabic Nasal (N - the "humming" sound).
    pub const fn syllabic_nasal(syn: TonedSyllabicNasal) -> Self {
        Self {
            kind: SyllableKind::SyllabicNasal(syn),
        }
    }

    /// Creates a Vowel Syllable (V).
    pub const fn vowel(vowel: TonedVowel) -> Self {
        Self {
            kind: SyllableKind::Vowel(vowel),
        }
    }

    /// Creates a Consonant Syllable (CV).
    pub const fn consonant(con: TonedConsonant) -> Self {
        Self {
            kind: SyllableKind::Consonant(con),
        }
    }

    /// Helper to create a CV syllable quickly.
    ///
    /// # Example
    /// ```rust
    /// use yoruba::{Syllable, Consonant, OralVowel, Tone};
    /// // "Bá"
    /// let s = Syllable::consonant_vowel(Consonant::Bi, OralVowel::Ah, Tone::High);
    /// ```
    pub const fn consonant_vowel(con: Consonant, oral: OralVowel, tone: Tone) -> Self {
        Self {
            kind: SyllableKind::Consonant(TonedConsonant(
                con,
                TonedVowel {
                    vowel: Vowel::Oral(oral),
                    tone,
                },
            )),
        }
    }

    /// Helper to create a C[NV] CV variant syllable quickly.
    ///
    /// # Example
    /// ```rust
    /// use yoruba::{Syllable, Consonant, NasalVowel, Tone};
    /// // "Yìn"
    /// let s = Syllable::consonant_nasal(Consonant::Yi, NasalVowel::In, Tone::Mid);
    /// ```
    pub const fn consonant_nasal(con: Consonant, nasal: NasalVowel, tone: Tone) -> Self {
        Self {
            kind: SyllableKind::Consonant(TonedConsonant(
                con,
                TonedVowel {
                    vowel: Vowel::Nasal(nasal),
                    tone,
                },
            )),
        }
    }

    /// Helper to create a Syllabic Nasal N.
    pub const fn n(tone: Tone) -> Self {
        Self {
            kind: SyllableKind::SyllabicNasal(TonedSyllabicNasal {
                tone,
                syllabic_nasal: SyllabicNasal::Hn,
            }),
        }
    }

    /// Helper to create a Syllabic Nasal M.
    pub const fn m(tone: Tone) -> Self {
        Self {
            kind: SyllableKind::SyllabicNasal(TonedSyllabicNasal {
                tone,
                syllabic_nasal: SyllabicNasal::Hm,
            }),
        }
    }

    /// Helper to create a Oral-Vowel-only syllable.
    pub const fn oral_vowel(oral: OralVowel, tone: Tone) -> Self {
        Self {
            kind: SyllableKind::Vowel(TonedVowel {
                tone,
                vowel: Vowel::Oral(oral),
            }),
        }
    }

    /// Helper to create a Nasal-Vowel-only syllable (e.g. `Un`).
    pub const fn nasal_vowel(nasal: NasalVowel, tone: Tone) -> Self {
        Self {
            kind: SyllableKind::Vowel(TonedVowel {
                tone,
                vowel: Vowel::Nasal(nasal),
            }),
        }
    }

    pub const fn kind(self) -> SyllableKind {
        self.kind
    }
}

impl fmt::Display for Syllable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            SyllableKind::Vowel(tv) => write!(f, "{}", tv),
            SyllableKind::Consonant(TonedConsonant(consonant, vowel)) => {
                write!(f, "{}{}", consonant.as_str(), vowel)
            }
            SyllableKind::SyllabicNasal(sn) => write!(f, "{}", sn),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tone_marks() {
        assert_eq!(Tone::Mid.mark(), None);
        assert_eq!(Tone::Low.mark(), Some('\u{0300}'));
        assert_eq!(Tone::High.mark(), Some('\u{0301}'));
    }

    #[test]
    fn test_vowel_display() {
        // High Tone A -> Á
        let v = TonedVowel {
            vowel: Vowel::Oral(OralVowel::Ah),
            tone: Tone::High,
        };
        assert_eq!(std::format!("{}", v), "á");

        // Low Tone E-Dot -> Ẹ̀
        let v = TonedVowel {
            vowel: Vowel::Oral(OralVowel::Eh),
            tone: Tone::Low,
        };
        assert_eq!(std::format!("{}", v), "ẹ̀");
    }

    #[test]
    fn test_syllable_construction() {
        // "Bá"
        let s = Syllable::consonant_vowel(Consonant::Bi, OralVowel::Ah, Tone::High);
        assert_eq!(std::format!("{}", s), "bá");

        // "Nǹ" (Nasal low)
        let s = Syllable::n(Tone::Low);
        assert_eq!(std::format!("{}", s), "ǹ");
    }

    #[test]
    fn test_complex_word_bimbola() {
        const AMBIMBOLA: [Syllable; 5] = [
            // A (V - Mid)
            Syllable::oral_vowel(OralVowel::Ah, Tone::Mid),
            // Bí (CV - High)
            Syllable::consonant_vowel(Consonant::Bi, OralVowel::I, Tone::High),
            // m (Syllabic Nasal - Mid)
            Syllable::m(Tone::Mid),
            // bọ́ (CV - High)
            Syllable::consonant_vowel(Consonant::Bi, OralVowel::Or, Tone::High),
            // lá (CV - High)
            Syllable::consonant_vowel(Consonant::Li, OralVowel::Ah, Tone::High),
        ];
        let syllables = AMBIMBOLA;
        let word: std::string::String = syllables.iter().map(|s| std::format!("{}", s)).collect();
        assert_eq!(word, "abímbọ́lá");
    }
}
