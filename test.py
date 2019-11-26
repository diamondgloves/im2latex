from img2latex import load_model,img2latex
if __name__ == '__main__':
    img_path = 'data/images_test/1.png'
    model = load_model()
    tex_seq = img2latex(model, img_path)
    print(tex_seq)