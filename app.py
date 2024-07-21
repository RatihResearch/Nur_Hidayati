from DataCleansing import Preprocessing
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr

model_checkpoint = 'finetune-indosum'
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

def get_summary(article):
    try:            
        preprocessor = Preprocessing(article)
        cleaned_berita = preprocessor._process_text()
        input_ids = tokenizer.encode(cleaned_berita, return_tensors='pt')
        summary_ids = model.generate(input_ids, 
                                          max_length=100,
                                          num_beams=2,
                                          repetition_penalty=2.5,
                                          length_penalty=1.0,
                                          early_stopping=True,
                                          no_repeat_ngram_size=2,
                                          use_cache=True)
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
        return summary_text
    
    except:
        return "error"
    
with gr.Blocks(title ="PENERAPAN METODE DEEP LEARNING UNTUK PERINGKASAN BERITA OTOMATIS PADA APLIKASI PETAKABAR") as demo:

    gr.Markdown(
     """
     # <center> PENERAPAN METODE DEEP LEARNING UNTUK PERINGKASAN BERITA OTOMATIS PADA APLIKASI PETAKABAR 
     <center> Tool sederhana untuk membuat ringkasan dari sebuah artikel atau berita.
     """
     )
    with gr.Tab("Labelling Text"):
         gr.Interface(
              fn=get_summary,
              inputs=gr.Textbox(lines=13, placeholder="Masukkan teks berita disini", label="Input Berita"),
              outputs=gr.Textbox(lines=3, label="Summary"),
              allow_flagging="never"
         )

if __name__ == "__main__":
    demo.launch()