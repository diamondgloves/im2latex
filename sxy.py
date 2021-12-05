from os.path import join
from model.utils.text import Vocab, load_formulas
from model.utils.image import greyscale, build_images

if __name__ == '__main__':
    formula = r'( a _ { i j } ) = \left( \begin{matrix} { 2 } & { - 1 } \\ { - 1 } & { 0 } \\ \end{matrix} \right) .'
    with open("{}.tex".format('sxy'), "w") as f:
        f.write(
    r"""\documentclass[preview]{standalone}
    \usepackage{amsmath}
    \begin{document}
        \[ %s \]
    \end{document}""" % formula)

    # formula_ref = "formula_ref.txt"
    # images_ref = "images_ref"
    # build_images(load_formulas(formula_ref), images_ref)