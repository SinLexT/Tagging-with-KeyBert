import streamlit as st
from keybert import KeyBERT
from transformers import TFBertModel
from unicodedata import normalize

def replace_ptbr_char_by_word(word):
    """ Will remove the encode token by token"""
    word = str(word)
    word = normalize('NFKD', word).encode('ASCII', 'ignore').decode('ASCII')
    return word

def remove_pt_br_char_by_text(text, stop_words):
    """ Will remove the encode using the entire text"""
    text = str(text)
    text = " ".join(replace_ptbr_char_by_word(word) for word in text.split() if word not in stop_words)
    return text

def keybert(doc):
    with open('stopwords_id.txt', 'r') as f:
        yake_stop_words = f.read().split()

    text = remove_pt_br_char_by_text(doc, yake_stop_words)

    custom_kw_model = KeyBERT(TFBertModel.from_pretrained("indobenchmark/indobert-large-p2"))
    keywords = custom_kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3),
                                                use_mmr=True, diversity=0.2,
                                                stop_words=None, top_n=15)
    
    list_keyword = []
    for kw in keywords:
        list_keyword.append(kw[0])
    return list_keyword

def main():
    st.title('Tag Recommendations')
    text = st.text_input("Text", "Type Here")

    # clean text
    text = text.replace('Terima kasih atas pertanyaan Anda.', '')
    text = text.replace('Artikel di bawah ini adalah pemutakhiran dari artikel dengan judul', '')
    text = text.replace('Bingung menentukan keterkaitan pasal dan kewajiban bisnis Anda, serta keberlakuan peraturannya? Ketahui kewajiban dan sanksi hukum perusahaan Anda dalam satu platform integratif dengan Regulatory Compliance System dari Hukumonline, klik di sini untuk mempelajari lebih lanjut', '')
    text = text.replace('Seluruh informasi hukum yang ada di Klinik hukumonline.com disiapkan semata – mata untuk tujuan pendidikan dan bersifat umum (lihat Pernyataan Penyangkalan selengkapnya). Untuk mendapatkan nasihat hukum spesifik terhadap kasus Anda, konsultasikan langsung dengan Konsultan Mitra Justika.', '')
    tag_result = ''
    final_result = []
    stopwords = ['pasal', 'nomor', 'ayat', 'undang', 'angka',
                 'undang nomor', 'pemerintah nomor', 'hukum',
                 'tahun', 'bangsa', 'negara', 'republik',
                 'indonesia', 'nasional', 'adalah', 'huruf',
                 'undang-undang nomor', 'untuk']
    if st.button("Give Recommendations"):
        tag_result = keybert(text.lower())
        for i in tag_result:
            if i not in stopwords:
                # words = i.split()
                # result_word = [word for word in words if word.lower() not in kata_hubung]
                # result_keyword = ' '.join(result_word)
                final_result.append(i)
    st.success("Tag recommendations: {}".format(final_result))


if __name__ == '__main__':
    main()
