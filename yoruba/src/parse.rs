//! # Yoruba Text Parser
//!
//! This module implements a recursive descent parser for the Yoruba language.
//!
//! ## Parsing Pipeline
//!
//! The parsing process is layered to handle Unicode complexity and Yoruba orthography rules:
//!
//! 1.  **Normalization & Atomization (`YorubaAtoms`)**:
//!     Raw text is first normalized to NFD (Normalization Form Decomposition). This splits
//!     characters like `ẹ́` into `e` + `dot_below` + `acute`. The stream is then tokenized
//!     into "Atoms" (Letters, Accents, Dots, Apostrophes).
//!
//! 2.  **Character Reconstruction (`YorubaChars`)**:
//!     Atoms are greedily consumed to reconstruct full Yoruba characters (Base + Tone + Dot).
//!     This abstraction handles the chaotic ordering of combining marks (e.g., handling
//!     both `e + dot + acute` and `e + acute + dot` identically).
//!
//! 3.  **Syllabic Parsing (`Parser` & Type impls)**:
//!     The stream of `YorubaChar`s is analyzed to build high-level phonological units:
//!     -   **Consonants**: (e.g., `b`, `gb`, `ṣ`).
//!     -   **Vowels**: Oral (`a`, `e`) and Nasal (`an`, `un`).
//!     -   **Tones**: Low (Grave), Mid (Null/Macron), High (Acute).
//!
//! ## Key Challenges Handled
//!
//! -   **Ambiguity of 'N'**: The parser distinguishes between 'n' as a consonant (`n-i`),
//!     'n' as a nasal marker (`i-n`), and 'n' as a syllabic nasal (`n-la`).
//! -   **Elision**: Handles apostrophes used in elision (e.g., `n'le`).
//! -   **Diacritic Noise**: Filters out or correctly interprets various Unicode representations
//!     of tones and dots.

use core::{
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    str::Chars,
};

use crate::{
    Consonant, NasalVowel, OralVowel, SyllabicNasal, Syllable, Tone, TonedConsonant,
    TonedSyllabicNasal, TonedVowel, Vowel,
};

use AccentAtom::*;
use LetterAtom::*;
use YorubaAtom::*;
use unicode_normalization::{Decompositions, UnicodeNormalization as _};

// ============================================================================
// REGION: CONSTANTS & HELPERS
// ============================================================================

pub(crate) const COM_GRAVE: char = '\u{0300}'; // Do (Low Tone)
pub(crate) const COM_MACRON: char = '\u{0304}'; // Re (Mid Tone - explicit)
pub(crate) const COM_TILDE: char = '\u{0303}'; // Sometimes used for Mid tone
pub(crate) const COM_ACUTE: char = '\u{0301}'; // Mi (High Tone)
pub(crate) const COM_DOT_BELOW: char = '\u{0323}';
pub(crate) const COM_LINE_BELOW: char = '\u{0329}'; // Often used interchangeably with dot below

/// Helper function to compose a base character with a tone.
/// Prefers precomposed Unicode chars (NFC) where available.
pub(crate) const fn compose_yoruba_lower(base: char, tone: Tone) -> Option<char> {
    match (base, tone) {
        // A
        ('a', Tone::High) => Some('\u{00E1}'), // á
        ('a', Tone::Low) => Some('\u{00E0}'),  // à
        ('a', Tone::Mid) => Some('a'),         // a
        // E
        ('e', Tone::High) => Some('\u{00E9}'), // é
        ('e', Tone::Low) => Some('\u{00E8}'),  // è
        ('e', Tone::Mid) => Some('e'),         // e
        // I
        ('i', Tone::High) => Some('\u{00ED}'), // í
        ('i', Tone::Low) => Some('\u{00EC}'),  // ì
        ('i', Tone::Mid) => Some('i'),         // i
        // O
        ('o', Tone::High) => Some('\u{00F3}'), // ó
        ('o', Tone::Low) => Some('\u{00F2}'),  // ò
        ('o', Tone::Mid) => Some('o'),         // o
        // U
        ('u', Tone::High) => Some('\u{00FA}'), // ú
        ('u', Tone::Low) => Some('\u{00F9}'),  // ù
        ('u', Tone::Mid) => Some('u'),         // u
        // N
        ('n', Tone::High) => Some('\u{0144}'), // ń
        ('n', Tone::Low) => Some('\u{01F9}'),  // ǹ
        ('n', Tone::Mid) => Some('n'),         // n
        // M
        ('m', Tone::High) => Some('\u{1E3F}'), // ḿ
        // m-grave isn't always precomposed in all fonts, so we return None to let
        // the caller fall back to combining marks.
        _ => None,
    }
}

// ============================================================================
// STEP 1: LOW-LEVEL ATOMIZER
// ============================================================================

/// Represents the absolute smallest units of Yoruba text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum YorubaAtom {
    /// A fundamental letter (A-Z), stripped of all diacritics.
    Letter(LetterAtom),
    /// A tone mark.
    Accent(AccentAtom),
    /// A sub-dot (or vertical line below) indicating open vowel quality or 'S' modification.
    Dot,
    /// An apostrophe, crucial for parsing elided words like `n'le`.
    Apostrophe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LetterAtom {
    A,
    B,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    R,
    S,
    T,
    U,
    W,
    Y,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccentAtom {
    Do, // Low
    Re, // Mid (often implicit)
    Mi, // High
}

/// Tokenizes text into Atoms, handling Unicode decomposition.
///
/// This ensures `ẹ` (one char) and `e` + `dot` (two chars) produce the same atom stream.
#[derive(Clone)]
pub struct YorubaAtoms<'a> {
    decompose: Decompositions<Chars<'a>>,
}

impl<'a> YorubaAtoms<'a> {
    pub fn from_str(s: &'a str) -> Self {
        Self {
            decompose: s.nfkd(),
        }
    }
}

impl Iterator for YorubaAtoms<'_> {
    type Item = YorubaAtom;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // FIXME: We might need to peek to make the error sticky on
        // no recognized char since this is public... but that's expensive
        let ch = self.decompose.next()?;
        Some(match ch {
            // --- Letters (Case Insensitive) ---
            'A' | 'a' => Letter(A),
            'b' | 'B' => Letter(B),
            'd' | 'D' => Letter(D),
            'e' | 'E' => Letter(E),
            'f' | 'F' => Letter(F),
            'g' | 'G' => Letter(G),
            'h' | 'H' => Letter(H),
            'i' | 'I' => Letter(I),
            'j' | 'J' => Letter(J),
            'k' | 'K' => Letter(K),
            'l' | 'L' => Letter(L),
            'm' | 'M' => Letter(M),
            'n' | 'N' => Letter(N),
            'o' | 'O' => Letter(O),
            'p' | 'P' => Letter(P),
            'r' | 'R' => Letter(R),
            's' | 'S' => Letter(S),
            't' | 'T' => Letter(T),
            'u' | 'U' => Letter(U),
            'w' | 'W' => Letter(W),
            'y' | 'Y' => Letter(Y),

            // --- Tones ---
            COM_GRAVE => Accent(Do),
            COM_TILDE | COM_MACRON => Accent(Re),
            COM_ACUTE => Accent(Mi),

            // --- Modification Dots ---
            COM_DOT_BELOW | COM_LINE_BELOW => Dot,

            // --- Punctuation used in Grammar ---
            '\'' | '’' => Apostrophe,

            // Discard anything else (e.g., random punctuation, emojis).
            // XXX: Should we error
            _ => return None,
        })
    }
}

// ============================================================================
// STEP 2: CHARACTER RECONSTRUCTION
// ============================================================================

/// A fully reconstructed Yoruba character.
///
/// This could represent a base letter plus any modifiers that apply to it or an apostrophe.
/// For example, `ẹ́` is represented as:
/// - `Base`: E
/// - `Dot`: true
/// - `Accent`: Some(Mi) (High Tone)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct YorubaChar {
    base_character: LetterAtom,
    accent: Option<AccentAtom>,
    has_dot: bool,
}

impl YorubaChar {
    // Could be apostrophe even though not handled now
    #[inline]
    pub const fn as_real_character_parts(self) -> Option<(LetterAtom, Option<AccentAtom>, bool)> {
        Some((self.base_character, self.accent, self.has_dot))
    }
}

#[derive(Clone)]
pub struct YorubaChars<'a> {
    atoms: core::iter::Peekable<YorubaAtoms<'a>>,
}

impl<'a> YorubaChars<'a> {
    pub fn from_str(s: &'a str) -> Self {
        Self {
            atoms: YorubaAtoms::from_str(s).peekable(),
        }
    }
}

impl Iterator for YorubaChars<'_> {
    type Item = YorubaChar;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        next_char(&mut self.atoms)
    }
}

/// The "Greedy Consumer" logic.
///
/// This function grabs a base letter, then looks ahead to consume as many valid
/// modifiers (dots, accents) as possible. This handles the ambiguity of
/// modifier order in Unicode (e.g., whether the dot comes before or after the acute accent).
#[inline]
fn next_char(atoms: &mut core::iter::Peekable<YorubaAtoms<'_>>) -> Option<YorubaChar> {
    // 1. Get the base character.
    let first_atom = atoms.next()?;

    // 2. Initialize the struct.
    let mut y_char = match first_atom {
        YorubaAtom::Letter(l) => YorubaChar {
            base_character: l,
            accent: None,
            has_dot: false,
        },
        // If we hit a floating diacritic or apostrophe at the start, it's invalid.
        // We return None, effectively stopping the iterator.
        // FIXME: Not really, once we handle apostrophe more gracefully.
        _ => return None,
    };

    // 3. Greedy Lookahead Loop.
    // We loop because they can appear in any order (e.g. e + dot + acute).
    loop {
        match atoms.peek() {
            Some(YorubaAtom::Dot) => {
                // Found a dot. Mark it and consume.
                //
                // If we already have a dot, this is a double dot (invalid?),
                // but we consume it to avoid getting stuck.
                y_char.has_dot = true;
                atoms.next();
            }
            Some(YorubaAtom::Accent(acc)) => {
                // Found an accent. Set it and consume.
                // XXX: If multiple accents appear, the last one wins.
                y_char.accent = Some(*acc);
                atoms.next();
            }
            // If it's a letter or apostrophe, that belongs to the *next* character.
            _ => break,
        }
    }

    Some(y_char)
}

// ============================================================================
// STEP 3: PARSER INFRASTRUCTURE & RESOLVER
// ============================================================================

// --- Ambiguity Resolution ---

/// Context provided to the ambiguity resolver.
#[derive(Debug, Clone)]
pub struct NasalAmbiguityContext<'a> {
    base_vowel: YorubaChar,
    next_atom: YorubaAtom,
    remaining_atoms: core::iter::Peekable<YorubaAtoms<'a>>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> NasalAmbiguityContext<'a> {
    /// The vowel character immediately preceding the 'n'.
    pub fn base_vowel(&self) -> YorubaChar {
        self.base_vowel
    }

    /// The atom immediately following the 'n' (typically a Consonant or Dot).
    pub fn next_atom(&self) -> YorubaAtom {
        self.next_atom
    }

    // Get the rest of the atoms
    pub fn remaining_atoms(self) -> impl Iterator<Item = YorubaAtom> {
        // into_inner or somn is unstable for Peekable
        self.remaining_atoms
    }
}

/// The outcome of an ambiguity resolution decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NasalAmbiguityResolution {
    /// Treat the 'N' as a Nasal Marker attached to the previous vowel (e.g., `ran`).
    NasalVowel,
    /// Treat the 'N' as a standalone Syllabic Nasal (e.g., `Ba-n...`).
    SyllabicNasal,
}

/// A strategy trait for resolving the "Ambiguous N" problem.
///
/// This allows the parser to delegate linguistic decisions (like "Is 'Bandele' one or two syllables?")
/// to an external system, such as a dictionary or user preference, without hardcoding
/// exceptions into the parser logic.
///
/// Call decides how to parse an 'N' that follows a vowel and precedes a consonant/EOS.
pub type NasalAmbiguityResolver = dyn Fn(NasalAmbiguityContext) -> Option<NasalAmbiguityResolution>;

/// The default strategy: "Greedy Nasalization".
///
/// This assumes that any `Vowel + N + Consonant` sequence forms a Nasal Vowel.
/// This works for 99% of standard Yoruba words (`ranti`, `fún`, `nǹkan`) and
/// dialectal tone preservation (`Ògèdèngbé`).
/// It fails only on compound contractions like `Bandele` (parsing it as `Ban-de-le`),
/// which is acoustically acceptable.
pub const STANDARD_NASAL_RESOLVER: &'static NasalAmbiguityResolver =
    &|_ctx| Some(NasalAmbiguityResolution::NasalVowel);

/// The internal state machine for the parser.
///
/// It wraps a `Peekable` iterator of `YorubaAtoms`. We use atoms (not chars)
/// as the underlying stream because sometimes we need to peek at specific
/// atomic components (like checking if an 'n' is followed by a dot) before
/// fully committing to parsing a character.
#[derive(Clone)]
struct Parser<'a> {
    atoms: core::iter::Peekable<YorubaAtoms<'a>>,
    nasal_ambiguity_resolver: &'a NasalAmbiguityResolver,
}

impl<'a> Parser<'a> {
    fn from_str(s: &'a str) -> Self {
        Self {
            atoms: YorubaAtoms::from_str(s).peekable(),
            nasal_ambiguity_resolver: STANDARD_NASAL_RESOLVER,
        }
    }

    /// Convenience wrapper to pull a full `YorubaChar` from the stream.
    /// NOTE: This advances the stream. Use `clone()` if you need to peek deep.
    #[inline]
    fn next_char(&mut self) -> Option<YorubaChar> {
        next_char(&mut self.atoms)
    }

    fn is_exhausted(&mut self) -> bool {
        // FIXME: There could be more even after the first None
        self.atoms.peek().is_none()
    }
}

// Allow direct access to the underlying iterator methods (peek, etc.)
impl DerefMut for Parser<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.atoms
    }
}
impl<'a> Deref for Parser<'a> {
    type Target = core::iter::Peekable<YorubaAtoms<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.atoms
    }
}

// ============================================================================
// REGION: PARSING LOGIC (THE CORE)
// ============================================================================

impl Tone {
    /// Parses a tone from the next atom.
    ///
    /// If no accent atom is present, this assumes `Tone::Mid` (the default in Yoruba).
    #[inline]
    fn parse_must(parser: &mut Parser) -> Self {
        match parser.peek() {
            Some(Accent(Do)) => {
                parser.next();
                Self::Low
            }
            Some(Accent(Mi)) => {
                parser.next();
                Self::High
            }
            // If it's a letter, dot, or apostrophe, it's not a tone.
            // Therefore, the previous segment had a Mid tone.
            _ => Self::Mid,
        }
    }

    #[inline]
    fn from_accent(acc: Option<AccentAtom>) -> Tone {
        match acc {
            Some(Do) => Tone::Low,
            Some(Mi) => Tone::High,
            Some(Re) | None => Tone::Mid,
        }
    }
}

impl TonedVowel {
    /// Parses a Vowel (Oral or Nasal) with its Tone.
    ///
    /// # The Logic of "N"
    /// This is the most complex part of the parser. In Yoruba, the letter 'n' usually
    /// indicates nasality when it follows a vowel (e.g., `an`, `on`), *unless* it
    /// is the onset of the *next* syllable (e.g., `a-ni`).
    #[inline]
    fn parse(parser: &mut Parser) -> Option<Self> {
        // 1. Parse the Base Vowel (e.g., 'a', 'e', 'o').
        let base = parser.next_char()?;

        // 2. Check for Nasality marker ('n').
        // We need to know if the NEXT char is 'n'.
        // AND if that 'n' is part of this syllable or the next.
        let starts_with_n = matches!(parser.peek(), Some(Letter(N)));

        let is_nasal = starts_with_n && {
            // DEEP LOOKAHEAD: Is this 'n' a nasal marker or start of next syllable?
            let mut lookahead = parser.clone();
            lookahead.next(); // Consume 'n'

            // What comes AFTER 'n'?
            match lookahead.next() {
                // RULE 1: The "Tone Guard"
                // If 'n' carries an explicit tone (High/Low), it is a Syllabic Nasal.
                // It CANNOT be part of this vowel.
                // e.g. "mòńlọ" -> "mò" + "ń" + "lọ"
                Some(Accent(_)) => false,

                // RULE 2: The "Syllable Onset"
                // If 'n' is followed by a vowel, it starts a new syllable.
                // e.g. "a-ni", "o-ni".
                Some(Letter(A | E | I | O | U)) => {
                    // FIXME: Actually... I think it might could be any of the 3
                    // possibilites e.g wọn-ó-sìlo (and they will go). Here, the n
                    // forms a part of this syllable.
                    // Maybe this isn't
                    // an acceptable way to write Yoruba... maybe it needs space.
                    false
                }

                // RULE 3: Ambiguous Zone (Consonant or End of String). Consult resolver
                Some(Letter(consonant)) => {
                    // It has come to this... the n has nothing on it,
                    // and we have consonant. e.g Ba-n-de-le (standalone), Ran-ti (merged backward), Ò-gè-dèn-gbé (special)
                    let cx = NasalAmbiguityContext {
                        // FIXME: Right here we could be very wrong about this
                        // being vowel since we did not check prior to now.
                        // Checking twice isn't that expensive but a more concise
                        // solution would be desired.
                        base_vowel: base,
                        next_atom: Letter(consonant),
                        remaining_atoms: lookahead.atoms,
                        _marker: PhantomData,
                    };

                    // None means they couldn't conclude either so error... it's
                    // wrong to default to true/false
                    match (parser.nasal_ambiguity_resolver)(cx)? {
                        NasalAmbiguityResolution::NasalVowel => true,
                        NasalAmbiguityResolution::SyllabicNasal => false,
                    }
                }

                // RULE 4: End of string/punctuation (e.g., "fún") 'n' is definitely
                // a nasal marker.
                _ => true,
            }
        };

        // 3. Construct the specific Vowel Enum based on nasality and base char.
        if is_nasal {
            parser.next(); // Consume 'n'

            // Map Base + N -> NasalVowel
            let nv = match base.base_character {
                A => NasalVowel::An,
                E if base.has_dot => NasalVowel::En, // ẹn
                I => NasalVowel::In,
                O if base.has_dot => NasalVowel::Orn, // ọn
                U => NasalVowel::Un,
                // Dialectal 'en' support or invalid
                #[cfg(feature = "dialects")]
                E => NasalVowel::Eyn,
                _ => return None,
            };

            Some(Self {
                vowel: Vowel::Nasal(nv),
                tone: Tone::from_accent(base.accent),
            })
        } else {
            // Map Base -> OralVowel
            let ov = match base.base_character {
                A => OralVowel::Ah,
                E if base.has_dot => OralVowel::Eh, // ẹ
                E => OralVowel::Ey,                 // e
                I => OralVowel::I,
                O if base.has_dot => OralVowel::Or, // ọ
                O => OralVowel::Oh,                 // o

                U => OralVowel::U,
                _ => return None,
            };

            Some(Self {
                vowel: Vowel::Oral(ov),
                tone: Tone::from_accent(base.accent),
            })
        }
    }
}

impl Consonant {
    /// Attempts to parse a Consonant.
    ///
    /// Handles digraphs (`gb`) and modified characters (`ṣ`).
    #[inline]
    fn parse(parser: &mut Parser) -> Option<Self> {
        Some(match parser.next()? {
            Letter(B) => Consonant::Bi,
            Letter(D) => Consonant::Di,
            Letter(F) => Consonant::Fi,
            Letter(G) => {
                // Lookahead: Is this 'g' or 'gb'?
                match parser.peek() {
                    Some(Letter(B)) => {
                        parser.next(); // Consume 'b'
                        Consonant::Gbi
                    }
                    _ => Consonant::Gi,
                }
            }
            Letter(H) => Consonant::Hi,
            Letter(J) => Consonant::Ji,
            Letter(K) => Consonant::Ki,
            Letter(L) => Consonant::Li,
            Letter(M) => Consonant::Mi,
            Letter(N) => Consonant::Ni,
            Letter(P) => Consonant::Pi,
            Letter(R) => Consonant::Ri,
            Letter(S) => {
                // Lookahead: Is this 's' or 'ṣ' (sh)?
                // 'ṣ' can be represented as S + Dot or S + H (in anglicized text).
                match parser.peek() {
                    Some(Letter(H) | Dot) => {
                        parser.next(); // Consume dot/h
                        Consonant::Shi
                    }
                    _ => Consonant::Si,
                }
            }
            Letter(T) => Consonant::Ti,
            Letter(W) => Consonant::Wi,
            Letter(Y) => Consonant::Yi,
            // Vowels or other atoms are not consonants.
            _ => return None,
        })
    }
}

impl TonedConsonant {
    /// Parses a CV (Consonant-Vowel) unit.
    #[inline]
    fn parse(parser: &mut Parser) -> Option<Self> {
        let con = Consonant::parse(parser)?;
        let vowel = TonedVowel::parse(parser)?;
        Some(TonedConsonant(con, vowel))
    }
}

impl TonedSyllabicNasal {
    /// Parses a Syllabic Nasal (a nasal sound that acts as a syllable nucleus).
    /// E.g., the `n` in `n-lò` or `m` in `m-bọ`.
    #[inline]
    fn parse(parser: &mut Parser) -> Option<Self> {
        // Note: We don't use `next_char` here because 'n' might not have a tone mark attached directly
        // in the atom stream yet, or might be followed by an apostrophe.

        let syllabic_nasal = match parser.next()? {
            Letter(N) => SyllabicNasal::Hn,
            Letter(M) => SyllabicNasal::Hm,
            _ => return None,
        };

        let tone = Tone::parse_must(parser);

        // Check for Elision Apostrophe (e.g., "n'le").
        // If present, we consume it. The 'n' is the syllabic unit, 'le' is the next syllable.
        if let Some(Apostrophe) = parser.peek() {
            parser.next();
        }

        Some(Self {
            syllabic_nasal,
            tone,
        })
    }
}

impl Syllable {
    /// The Main Parser Entry Point.
    ///
    /// Scans the stream and attempts to match the next sequence to a valid syllable structure.
    /// The order of attempts is significant:
    /// 1. **Consonant-Vowel (CV)**: The most common structure.
    /// 2. **Vowel Only (V)**: Often prefixes.
    /// 3. **Syllabic Nasal (N)**: 'n' or 'm' standing alone.
    fn parse(parser: &mut Parser) -> Option<Self> {
        // We clone `parser` to peek deeply. If parsing fails mid-way, we revert.

        // Priority 1: CV (Consonant + Vowel)
        let mut fork = parser.clone();
        if let Some(con) = TonedConsonant::parse(&mut fork) {
            *parser = fork; // Commit
            return Some(Self::consonant(con));
        }

        // Priority 2: V (Vowel Only)
        let mut fork = parser.clone();
        if let Some(vowel) = TonedVowel::parse(&mut fork) {
            *parser = fork; // Commit
            return Some(Self::vowel(vowel));
        }

        // Priority 3: N (Syllabic Nasal)
        let mut fork = parser.clone();
        if let Some(nasal) = TonedSyllabicNasal::parse(&mut fork) {
            *parser = fork; // Commit
            return Some(Self::syllabic_nasal(nasal));
        }

        // 4. Failure
        None
    }
}

// ============================================================================
// REGION: FRAGMENTATION (WORDS & PUNCTUATION)
// ============================================================================

// TODO: We're planning the sound_parse module and I think
// it'd also have its version of this... trying to abstract
// could lead to friction and underperformance
/// Represents a discrete part of a sentence.
///
/// Preserving non-word fragments allows for accurate TTS timing and
/// reconstruction of the original text.
#[derive(Debug, Clone)]
#[non_exhaustive] // It is exhaustive but I most likely can't cover all
pub enum SentenceFragment<'a> {
    Word(WordSentenceFragment<'a>),
    /// A sequence of whitespace characters.
    Space(SpaceSentenceFragment<'a>),
    /// A punctuation mark (e.g., '.', '!', '?').
    Punctuation(PunctuationSentenceFragment<'a>),
}

impl Display for SentenceFragment<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Word(w) => Display::fmt(w, f),
            Self::Space(s) => Display::fmt(s, f),
            Self::Punctuation(p) => Display::fmt(p, f),
        }
    }
}

// TODO: Rename ParseSyllables?
pub type WordSentenceFragment<'a> = ParseSyllables<'a>;

// Or we could enumerate space
// TODO: Rename ParseSpace?
#[derive(Debug, Clone, PartialEq)]
pub struct SpaceSentenceFragment<'a>(&'a str);

impl Display for SpaceSentenceFragment<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Oh no, word logic is growing big, should we have another module?
// TODO: Rename ParsePunctuation?
#[derive(Debug, Clone, PartialEq)]
pub struct PunctuationSentenceFragment<'a>(&'a str);

impl Display for PunctuationSentenceFragment<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ============================================================================
// REGION: HIGH-LEVEL ITERATORS
// ============================================================================

/// An iterator over words and their surrounding context in a sentence.
#[derive(Clone)]
pub struct ParseWords<'a> {
    input: &'a str,
    cursor: usize,
    nasal_resolver: &'a NasalAmbiguityResolver,
}

impl<'a> ParseWords<'a> {
    /// Sets a custom resolution function for nasal ambiguity.
    pub fn with_nasal_resolver(mut self, resolver: &'a NasalAmbiguityResolver) -> Self {
        self.nasal_resolver = resolver;
        self
    }
}

impl<'a> Iterator for ParseWords<'a> {
    type Item = SentenceFragment<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        // FIXME: PROBLEMATIC... We could get something like
        // "olùké.è.kó." which is a valid word according to the
        // normalization rules
        if self.cursor >= self.input.len() {
            return None;
        }

        let remaining = &self.input[self.cursor..];
        let first_char = remaining.chars().next()?;

        // Mode 1: Whitespace
        if first_char.is_whitespace() {
            let len = remaining
                .find(|c: char| !c.is_whitespace())
                .unwrap_or(remaining.len());
            let space_slice = &remaining[..len];
            self.cursor += len;
            return Some(SentenceFragment::Space(SpaceSentenceFragment(space_slice)));
        }

        // Mode 2: Punctuation (excluding apostrophe usually handled in words)
        if first_char.is_ascii_punctuation() && first_char != '\'' {
            // NOTE: Apostrophe is treated as part of a word in Yoruba logic often,
            // but if it appears start-of-word unrelated to elision it might be tricky.
            // For now, we yield punctuation singly.
            let end = first_char.len_utf8();
            let punct_slice = &remaining[..end];
            self.cursor += end;
            return Some(SentenceFragment::Punctuation(PunctuationSentenceFragment(
                punct_slice,
            )));
        }

        // Mode 3: Word
        let len = remaining
            .find(|c: char| c.is_whitespace() || (c.is_ascii_punctuation() && c != '\''))
            .unwrap_or(remaining.len());

        let word_slice = &remaining[..len];
        self.cursor += len;

        Some(SentenceFragment::Word(
            parse_syllables(word_slice).with_nasal_resolver(self.nasal_resolver),
        ))
    }
}

impl Display for ParseWords<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let clone = self.clone();
        for fragment in clone {
            Display::fmt(&fragment, f)?;
        }
        Ok(())
    }
}

// ============================================================================
// PUBLIC ENTRY POINTS
// ============================================================================

// TODO: Rename parse_word?
/// Creates an iterator over syllables in a string.
///
/// This does not handle whitespace splitting; it assumes the input is a single word
/// or a sequence of syllables without breaks. Use `parse_words` for sentences.
#[inline]
pub fn parse_syllables<I: AsRef<str> + ?Sized>(input: &I) -> ParseSyllables<'_> {
    ParseSyllables {
        parser: Parser::from_str(input.as_ref()),
    }
}

/// Iterator for parsing syllables.
#[derive(Clone)]
pub struct ParseSyllables<'a> {
    parser: Parser<'a>,
}

impl<'a> ParseSyllables<'a> {
    /// Sets a custom resolution function for handling 'N' ambiguities.
    pub fn with_nasal_resolver(mut self, resolver: &'a NasalAmbiguityResolver) -> Self {
        self.parser.nasal_ambiguity_resolver = resolver;
        self
    }
}

impl<'a> Iterator for ParseSyllables<'a> {
    type Item = Syllable;

    fn next(&mut self) -> Option<Self::Item> {
        self.parser.peek()?; // Ensure we aren't exhausted
        Syllable::parse(&mut self.parser)
    }
}

impl Display for ParseSyllables<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for syllable in self.clone() {
            Display::fmt(&syllable, f)?;
        }
        Ok(())
    }
}

/// Creates an iterator over word fragments in a sentence.
#[inline]
pub fn parse_words<I: AsRef<str> + ?Sized>(input: &I) -> ParseWords<'_> {
    ParseWords {
        nasal_resolver: STANDARD_NASAL_RESOLVER,
        input: input.as_ref(),
        cursor: 0,
    }
}

// ============================================================================
// REGION: BOILERPLATE, MACROS & ERRORS
// ============================================================================

macro_rules! implement_debug_for_iter {
    ($ty:ident) => {
        impl Debug for $ty<'_> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, concat!(stringify!($ty), "("))?;
                f.debug_list().entries(self.clone()).finish()?;
                write!(f, ")")?;
                Ok(())
            }
        }
    };
}

implement_debug_for_iter! {YorubaAtoms}
implement_debug_for_iter! {YorubaChars}
implement_debug_for_iter! {ParseSyllables}
implement_debug_for_iter! {ParseWords}

macro_rules! implement_from_str {
    ($ty:ident => $error_name:ident) => {
        // I don't think we will ever be able to compare values of this
        #[derive(Debug)]
        pub struct $error_name {
            _priv: (),
        }

        impl core::str::FromStr for $ty {
            type Err = $error_name;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let mut parser = Parser::from_str(s);
                let Some(me) = Self::parse(&mut parser) else {
                    return Err($error_name { _priv: () });
                };

                // Strict Parse: Ensure the entire string was consumed.
                if !parser.is_exhausted() {
                    return Err($error_name { _priv: () });
                }

                Ok(me)
            }
        }
    };
}

implement_from_str! { Consonant => ConsonantError }
implement_from_str! { TonedConsonant => TonedConsonantError }
implement_from_str! { TonedVowel => TonedVowelError }
implement_from_str! { TonedSyllabicNasal => TonedSyllabicNasalError }
implement_from_str! { Syllable => SyllableError }

// ============================================================================
// REGION: TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use core::str::FromStr as _;
    use std::{
        format,
        string::{String, ToString as _},
        vec::Vec,
    };

    use crate::OralVowel;

    use super::*;

    // --- Test Helpers ---

    macro_rules! test_vowel {
        ($input:expr, $expected_vowel:expr, $expected_tone:expr) => {
            let res = TonedVowel::from_str($input)
                .expect(&format!("Failed to parse input: {:?}", $input));
            assert_eq!(
                res.vowel,
                $expected_vowel.into(),
                "Vowel mismatch for input: {:?}",
                $input
            );
            assert_eq!(
                res.tone, $expected_tone,
                "Tone mismatch for input: {:?}",
                $input
            );
        };
    }

    macro_rules! test_consonant {
        ($input:expr, $expected_constant:expr) => {
            let res =
                Consonant::from_str($input).expect(&format!("Failed to parse input: {:?}", $input));
            assert_eq!(
                res, $expected_constant,
                "Consonant mismatch for input: {:?}",
                $input
            );
        };
    }

    macro_rules! test_syllables {
        ($input:expr, $expected_syllables:expr) => {
            let syllables: Vec<String> = parse_syllables($input).map(|s| s.to_string()).collect();
            assert_eq!(syllables, $expected_syllables);
        };
    }

    macro_rules! test_words {
        ($input:expr, $expected_syllables:expr) => {
            let words: Vec<String> = parse_words($input).map(|s| s.to_string()).collect();
            assert_eq!(words, $expected_syllables);
        };
    }

    // --- Vowel Parsing Tests ---

    #[test]
    fn test_simple_mid_tone() {
        test_vowel!("a", OralVowel::Ah, Tone::Mid);
        test_vowel!("e", OralVowel::Ey, Tone::Mid);
        test_vowel!("i", OralVowel::I, Tone::Mid);
        test_vowel!("o", OralVowel::Oh, Tone::Mid);
        test_vowel!("u", OralVowel::U, Tone::Mid);
    }
    #[test]
    fn test_simple_casing() {
        test_vowel!("A", OralVowel::Ah, Tone::Mid);
        test_vowel!("E", OralVowel::Ey, Tone::Mid);
    }

    #[test]
    fn test_dotted_vowels_precomposed() {
        // Ẹ (Eh) and Ọ (Or) as single chars without tone
        test_vowel!("ẹ", OralVowel::Eh, Tone::Mid);
        test_vowel!("Ẹ", OralVowel::Eh, Tone::Mid);
        test_vowel!("ọ", OralVowel::Or, Tone::Mid);
        test_vowel!("Ọ", OralVowel::Or, Tone::Mid);
    }

    #[test]
    fn test_precomposed_high_tones() {
        test_vowel!("á", OralVowel::Ah, Tone::High);
        test_vowel!("é", OralVowel::Ey, Tone::High);
        test_vowel!("í", OralVowel::I, Tone::High);
        test_vowel!("ó", OralVowel::Oh, Tone::High);
        test_vowel!("ú", OralVowel::U, Tone::High);
    }

    #[test]
    fn test_precomposed_low_tones() {
        test_vowel!("à", OralVowel::Ah, Tone::Low);
        test_vowel!("è", OralVowel::Ey, Tone::Low);
        test_vowel!("ì", OralVowel::I, Tone::Low);
        test_vowel!("ò", OralVowel::Oh, Tone::Low);
        test_vowel!("ù", OralVowel::U, Tone::Low);
    }

    #[test]
    fn test_complex_dotted_with_tones_precomposed() {
        // These are specific Unicode points (e.g. \u{1EBF})
        test_vowel!("ẹ́", OralVowel::Eh, Tone::High); // Precomposed Eh + Acute
        test_vowel!("Ẹ́", OralVowel::Eh, Tone::High);
        test_vowel!("ẹ̀", OralVowel::Eh, Tone::Low); // Precomposed Eh + Grave
        test_vowel!("Ẹ̀", OralVowel::Eh, Tone::Low);

        test_vowel!("ọ́", OralVowel::Or, Tone::High);
        test_vowel!("Ọ́", OralVowel::Or, Tone::High);
        test_vowel!("ọ̀", OralVowel::Or, Tone::Low);
        test_vowel!("Ọ̀", OralVowel::Or, Tone::Low);
    }

    // --- Unicode Normalization Stress Tests ---

    #[test]
    fn test_nfd_chaos() {
        // 'e' + dot below (u0323)
        test_vowel!("e\u{0323}", OralVowel::Eh, Tone::Mid);

        // 'e' + 'dot below' (\u{0323}) + 'acute' (\u{0301})
        test_vowel!("e\u{0323}\u{0301}", OralVowel::Eh, Tone::High);

        // Non-canonical order: e + acute + dot_below
        // Standard unicode normalizers might hate this, but our "Char-Hunt" eats it up.
        test_vowel!("e\u{0301}\u{0323}", OralVowel::Eh, Tone::High);

        // Mixed Precomposed: ẹ (dot built-in) + acute
        test_vowel!("ẹ\u{0301}", OralVowel::Eh, Tone::High);

        // Mixed Precomposed Reverse: é (acute built-in) + dot_below
        test_vowel!("é\u{0323}", OralVowel::Eh, Tone::High);

        // 'O' + 'grave' (\u{0300}) + 'dot below' (\u{0323})
        // Note: Different order of diacritics should still work
        test_vowel!("O\u{0300}\u{0323}", OralVowel::Or, Tone::Low);

        // "ó" (precomposed acute) + Dot below (separate char)
        test_vowel!("ó\u{0323}", OralVowel::Or, Tone::High);

        // Some fonts/inputs use vertical line below (\u{0329}) instead of dot below (\u{0323})
        // e + vertical line below
        test_vowel!("e\u{0329}", OralVowel::Eh, Tone::Mid);
    }

    #[test]
    fn test_weird_inputs() {
        // Whitespace handling - currently parse_words splits them, but parse_syllables
        // will stop at spaces if passed directly.
        // If we use TonedVowel::from_str(" a "), it fails because of leading space.
        // But parse_syllables(" a ") should work if we skip whitespace?
        // Actually, parse_syllables uses YorubaAtoms which stops at space.
        // So parse_syllables(" a ") -> returns [], because first char is space -> None.
        let mut s = parse_syllables(" a ");
        assert!(s.next().is_none());

        // Correct usage is parse_words or trimming first
        let res = TonedVowel::from_str("a").unwrap();
        assert_eq!(res.vowel, OralVowel::Ah.into());
    }

    #[test]
    fn test_invalid_inputs() {
        // Empty
        assert!(matches!(TonedVowel::from_str(""), Err(_)));
        assert!(matches!(TonedVowel::from_str("   "), Err(_)));

        // Consonants
        assert!(matches!(TonedVowel::from_str("b"), Err(_)));
        assert!(matches!(TonedVowel::from_str("gb"), Err(_)));

        // Symbols
        assert!(matches!(TonedVowel::from_str("1"), Err(_)));
        assert!(matches!(TonedVowel::from_str("$"), Err(_)));
    }

    // --- Nasal & Syllabic Tests ---

    #[test]
    fn test_nasal_lookahead_ambiguity() {
        // "an" -> Nasal A (Mid)
        test_vowel!("an", NasalVowel::An, Tone::Mid);

        // "án" -> Nasal A (High)
        test_vowel!("án", NasalVowel::An, Tone::High);

        // "fun" -> Nasal U (Mid) (User logic maps F+un, but here we test pure vowel 'un' like in 'un-un')
        test_vowel!("un", NasalVowel::Un, Tone::Mid);
    }

    #[test]
    fn test_nasal_vs_oral_boundary() {
        // Case: "ani" -> 'a' (Oral) - 'ni'.
        // If we parsed 'an' as nasal, we'd have 'i' left, which is wrong.
        // We test Syllable iterator to verify the stream.
        test_syllables!("ani", &["a", "ni"]);

        // Case: "funi" -> 'f' + 'u' (Oral) + 'ni'.
        // TonedConsonant should parse 'fu' (Oral) not 'fun' (Nasal) because 'n' belongs to 'ni'.
        test_syllables!("funi", &["fu", "ni"]);

        // Case: "fun" -> fun (Nasal)
        test_syllables!("fun", &["fun"]);

        // Case: "funn" (Nasal U + Syllabic N?) -> "fun" + "n"
        test_syllables!("funn", &["fun", "n"]);

        // Case: "rún"
        test_syllables!("rún", &["rún"]);

        // Case: "rúna"
        test_syllables!("rúna", &["rú", "na"]);

        // Case: "rántí"
        test_syllables!("ránti", &["rán", "ti"]);

        test_syllables!("ìtàn", &["ì", "tàn"]);
    }

    #[test]
    fn test_syllabic_nasal_break() {
        // "n" by itself is a syllabic nasal, NOT a vowel.
        // TonedVowel should fail.
        assert!(TonedVowel::from_str("n").is_err());

        // SyllabicNasal should pass.
        let res = TonedSyllabicNasal::from_str("n").unwrap();
        assert_eq!(res.syllabic_nasal, SyllabicNasal::Hn);
        assert_eq!(res.tone, Tone::Mid);

        // "ń" (High tone N)
        let res = TonedSyllabicNasal::from_str("ń").unwrap();
        assert_eq!(res.syllabic_nasal, SyllabicNasal::Hn);
        assert_eq!(res.tone, Tone::High);

        // "m" + grave
        let res = TonedSyllabicNasal::from_str("m\u{0300}").unwrap();
        assert_eq!(res.syllabic_nasal, SyllabicNasal::Hm);
        assert_eq!(res.tone, Tone::Low);

        // "nlo" (is going) -> n(High) - lo(Mid)
        // Ensure 'n' is treated as Nasal Syllable, not start of 'nl...'
        let mut syllables = parse_syllables("ǹlọ");
        assert_eq!(syllables.next().unwrap(), Syllable::n(Tone::Low));
        assert_eq!(
            syllables.next().unwrap(),
            Syllable::consonant_vowel(Consonant::Li, OralVowel::Or, Tone::Mid)
        );

        // "ḿbọ̀" (is coming) -> n(High) - lo(Low)
        // Ensure 'n' is treated as Nasal Syllable, not start of 'nl...'
        let mut syllables = parse_syllables("ḿbọ̀");
        assert_eq!(syllables.next().unwrap(), Syllable::m(Tone::High));
        assert_eq!(
            syllables.next().unwrap(),
            Syllable::consonant_vowel(Consonant::Bi, OralVowel::Or, Tone::Low)
        );

        test_syllables!("nǹkan", &["n", "ǹ", "kan"]);
        test_syllables!("Bímbọ́lá", &["bí", "m", "bọ́", "lá"]);
        test_syllables!("nla", &["n", "la"]);
    }

    #[test]
    fn test_syllabic_structure() {
        // "Bà"
        assert_eq!(
            Syllable::from_str("Bà").unwrap(),
            Syllable::consonant_vowel(Consonant::Bi, OralVowel::Ah, Tone::Low)
        );
    }

    #[test]
    fn test_elision_handling() {
        // "n'le" -> "n" (syllabic) + "le" (CV)
        // The parser should handle the apostrophe in the nasal parse or skip it?
        // Current impl consumes apostrophe in SyllabicNasal parse.
        test_syllables!("ǹ'lé", &["ǹ", "lé"]);

        // Let's check "ǹ'le"
        test_syllables!("ǹ'le", &["ǹ", "le"]);
    }

    // --- Consonant Tests ---

    #[test]
    fn test_gb_and_sh() {
        // Digraph Gb
        test_consonant!("Gb", Consonant::Gbi);

        // S vs Sh
        test_consonant!("S", Consonant::Si);
        test_consonant!("ṣ", Consonant::Shi);
        test_consonant!("sh", Consonant::Shi);
        // Decomposed Sh (s + dot)
        test_consonant!("s\u{0323}", Consonant::Shi);

        // "ṣ" -> Shi
        test_syllables!("ṣa", &["ṣa"]);

        // "shé" -> Shi + Ey + Mi
        test_syllables!("shé", &["ṣé"]);

        // "s" + dot_below -> Shi (NFD style consonant)
        test_syllables!("s\u{0323}a", &["ṣa"]);

        test_syllables!("gbẹ́ṣe", &["gbẹ́", "ṣe"]);
    }

    // --- Integration Tests ---

    #[test]
    fn test_full_syllables() {
        // "Orí" (Head) -> O(Mid) - ri(High)
        test_syllables!("Orí", &["o", "rí"]);
        // "gbàgbé" (forget) -> gba(Low) - gbe(High)
        test_syllables!("gbàgbé", &["gbà", "gbé"]);
        test_syllables!("bàbá", &["bà", "bá"]);
        test_syllables!("Alágbàdo", &["a", "lá", "gbà", "do"]);
        test_syllables!("Gbogbo", &["gbo", "gbo"]);
    }

    #[test]
    fn test_full_sentences() {
        // "mo ń lọ" (I am going)
        test_words!("mo ń lọ", &["mo", " ", "ń", " ", "lọ"]);

        // "mo ń lọ." (punctuation handling)
        test_words!("mo ń lọ.", &["mo", " ", "ń", " ", "lọ", "."]);

        // "Ra, wa"
        test_words!("Ra, wa", &["ra", ",", " ", "wa"]);
    }

    #[test]
    fn test_standard_resolver_behavior() {
        // Standard behavior is "Greedy Nasal"
        test_syllables!("mońlọ", &["mo", "ń", "lọ"]); // Tone guard check
        test_syllables!("rántí", &["rán", "tí"]); // Nasal check

        #[cfg(feature = "dialects")]
        test_syllables!("Ògèdèngbé", &["ò", "gè", "dèn", "gbé"]); // Dialectal check

        // "Bandele" gets parsed as "Ban-de-le" by default
        test_syllables!("Bándélé", &["bán", "dé", "lé"]);
    }

    #[test]
    fn test_custom_function_resolver() {
        fn bandele_resolver(_ctx: NasalAmbiguityContext) -> Option<NasalAmbiguityResolution> {
            Some(NasalAmbiguityResolution::SyllabicNasal)
            // if ctx.word_context.contains("Bandele") {
            //     return NasalAmbiguityResolution::SyllabicNasal;
            // }
            // NasalAmbiguityResolution::NasalVowel
        }

        let syllables: Vec<String> = parse_syllables("Bandele")
            .with_nasal_resolver(&bandele_resolver)
            .map(|s| s.to_string())
            .collect();

        // With the custom fn, we force the split!
        assert_eq!(syllables, &["ba", "n", "de", "le"]);
    }

    #[test]
    fn test_multivowel() {
        // Ancient spelling of "earth"
        // I don't know what else to syllabize it as... I'm tired of
        // this my language
        test_syllables!("Aiyé", &["a", "i", "yé"]);

        // Jesus christ... another problem! At "ì",
        // We see n and the standard resolver says
        // "À-ìn-á".
        //
        // BTW, this word is weird... it's called "Ainan"
        // So the TTS would also struggle with this calling
        // it "Ainah"
        test_syllables!("Àìná", &["à", "ì", "ná"]);
    }

    #[test]
    fn test_punctuation_fragmentation() {
        let mut words = parse_words("Mo lọ.");
        // 1. "Mo" (Word)
        match words.next().unwrap() {
            SentenceFragment::Word(mut s) => assert_eq!(s.next().unwrap().to_string(), "mo"),
            _ => panic!("Expected Word"),
        }
        // 2. " " (Space)
        match words.next().unwrap() {
            SentenceFragment::Space(s) => assert_eq!(s.to_string(), " "),
            _ => panic!("Expected Space"),
        }
        // 3. "lọ" (Word)
        match words.next().unwrap() {
            SentenceFragment::Word(mut s) => assert_eq!(s.next().unwrap().to_string(), "lọ"),
            _ => panic!("Expected Word"),
        }
        // 4. "." (Punctuation)
        match words.next().unwrap() {
            SentenceFragment::Punctuation(p) => assert_eq!(p.to_string(), "."),
            _ => panic!("Expected Punctuation"),
        }
    }

    #[test]
    fn test_garbage_handling() {
        // Should return None for non-sound inputs
        assert!(matches!(TonedVowel::from_str("zi"), Err(_))); // z is not a consonant
        assert!(matches!(TonedConsonant::from_str("1"), Err(_)));

        // "E123ka" -> E, 1(Garb), 2(Garb), 3(Garb), ka
        // The syllabizer might fail on garbage, but the tokenizer passes it through.
        // Actually, parse_syllables will just stop parsing when it hits garbage.
        let mut syllables = parse_syllables("Ẹ123ka");
        assert_eq!(syllables.next().unwrap().to_string(), "ẹ");
        assert!(syllables.next().is_none());
    }
}
