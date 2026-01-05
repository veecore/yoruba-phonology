# Yoruba Phonology & Speech Technologies

> *"I don't know what it was, but I suddenly got the impression that it'd be easy to write a Yoruba TTS... soon I realized it's not as easy as I thought but I've not given up."*

## 🚀 The Mission

This project is a dedicated effort to build **usable, robust, and linguistically accurate** Speech-to-Text (STT) and Text-to-Speech (TTS) engines for the Yoruba language.

Unlike generic speech tools, this project is built from the ground up with a deep respect for Yoruba phonology—specifically its tonal nature, syllable structure, and orthography. The goal is to move beyond simple "text processing" into true **language understanding** to drive high-quality synthesis and recognition.

## 🏗 Project Structure

The project is organized as a Rust workspace with three primary components:

### 1. `yoruba` (The Core Library)

The heart of the project. This library handles the fundamental linguistics of the language. It is strictly typed and handles the heavy lifting of text analysis before any audio processing happens.

* **Status**: Active & Functional.
* **Key Features**:
  * **Recursive Descent Parser**: Converts raw Unicode text into structured phonological units.
  * **Syllabification**: Breaks words into `V`, `CV`, and `N` (Syllabic Nasal) units.
  * **Tone Awareness**: First-class support for High (`Mi`), Low (`Do`), and Mid (`Re`) tones.
  * **Ambiguity Resolution**: Intelligently handles the "Ambiguous N" (determining if `n` is a consonant, a nasal marker, or a syllable on its own).
  * **Unicode Normalization**: Automatically handles NFD/NFC quirks (e.g., combining dots and accents).

### 2. `tts` (Text-to-Speech)

*Planned.*
This crate will handle the "Front-end" of the synthesis pipeline. Its responsibility is **normalization and preparation**.

* **Role**: To bridge the gap between "wild" input formats (PDFs, raw text files, user input) and the strict input required by the `yoruba` core.
* **Pipeline Vision**:
  1. **Ingest**: Read text from various sources (PDF, txt, md).
  2. **Clean**: Remove non-Yoruba artifacts.
  3. **Parse**: Pass clean text to `yoruba` to get a stream of `Syllable`s.
  4. **Synthesize**: Map `Syllable` + `Tone` pairs to audio samples (concatenative or parametric).

### 3. `stt` (Speech-to-Text)

*Planned.*
This crate handles the audio signal processing and transcription.

* **Role**: To convert raw audio waves into an intermediate representation (IR) that the `yoruba` library can validate or format.
* **Pipeline Vision**:
  1. **Audio Processing**: FFT / Spectrogram generation.
  2. **Feature Extraction**: Detecting pitch (to identify Tone) and formants (to identify Vowels).
  3. **IR Generation**: Producing a stream of potential phonemes.
  4. **Reconstruction**: Using `yoruba` logic to assemble these into valid syllables and words.

## 🛠 Usage (Core Library)

To use the core phonology parser in your Rust code:

```rust
use yoruba::{parse_syllables, Tone, Syllable};

fn main() {
    let text = "Bímbọ́lá";
    
    // The parser handles normalization and segmentation automatically
    let syllables: Vec<Syllable> = parse_syllables(text).collect();
    
    // Output: [bí, m, bọ́, lá]
    for syllable in syllables {
        println!("{}", syllable);
    }
}

```

## 🔮 Roadmap

* [x] **Phonological Parser**: Robust text-to-syllable parsing.
* [x] **Unicode Handling**: Support for messy diacritic inputs.
* [ ] **Automatic Re-accenter**: A context-aware engine to restore missing tone marks and underdots to "unaccented" Yoruba text (e.g., converting "nigba yen" -> "nígbà yẹn").
* [ ] **Audio Synthesis**: Basic concatenative synthesis based on parsed syllables.
* [ ] **Pitch Detection**: Accurate tone recognition from audio input.

## 🤝 Contribution

This is a passion project aimed at solving a hard problem. If you are interested in Signal Processing (DSP), NLP, or Yoruba Linguistics, your help is welcome!