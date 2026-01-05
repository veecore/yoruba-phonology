use yoruba::{parse_syllables, parse_words};

fn main() {
    let sentence = "Àwọn olùkẹ́ ẹ̀ kọ́  ti wolé iléẹ̀ kó";
    let fragments = parse_words(sentence);
    println!("{fragments}");

    for syllable in parse_syllables("Alágbàdo") {
        println!("{syllable}")
    }
}
