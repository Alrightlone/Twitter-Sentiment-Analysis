import sys
# your path is correct, keep it
sys.path.append('/Users/alright/Desktop/project_text_classification/PlotNeuralNet') 
from pycore.tikzeng import *
from pycore.blocks  import *

def main():
    arch = [
        to_head( '..' ),
        to_cor(),
        to_begin(),

        # === 1. input layer (Sentence) ===
        # 🔴 modification point 1: Caption changed to (60, 20)
        # 🔴 modification point 2: depth changed to 10 (visually thinner, represents smaller dimension)
        to_input('input.png', width=6, height=10, name="temp"), 
        to_Conv("emb", 60, 20, offset="(0,0,0)", to="(0,0,0)", height=40, depth=10, width=2, caption="Embedding\\\\(60, 20)"),

        # === 2. parallel convolution layer (TextCNN core) ===
        # here 100 is the number of filters, keep it
        
        # middle: 4-gram (baseline) -> note: 4-gram is the most common and effective n-gram
        to_Conv("conv4", 57, 100, offset="(3,0,0)", to="(emb-east)", height=35, depth=20, width=4, caption="4-gram"),
        to_Pool("pool4", offset="(1,0,0)", to="(conv4-east)", height=35, depth=5, width=1), 

        # upper: 3-gram (Z-axis positive offset) -> note: 3-gram is the second most common and effective n-gram
        to_Conv("conv3", 58, 100, offset="(0,0,4)", to="(conv4-west)", height=38, depth=20, width=4, caption="3-gram"),
        to_Pool("pool3", offset="(1,0,0)", to="(conv3-east)", height=38, depth=5, width=1),

        # lower: 5-gram (Z-axis negative offset) -> note: 5-gram is the least common and effective n-gram   
        to_Conv("conv5", 56, 100, offset="(0,0,-4)", to="(conv4-west)", height=32, depth=20, width=4, caption="5-gram"),
        to_Pool("pool5", offset="(1,0,0)", to="(conv5-east)", height=32, depth=5, width=1),

        # === 3. Concat layer ===
        # result is 300 (100*3). visually depth=60, compared to input depth=10, it can be clearly seen that the features have increased
        to_Conv("concat", 300, 1, offset="(3,0,0)", to="(pool4-east)", height=10, depth=60, width=2, caption="Concat\\\\(300)"),
        
        # draw connection lines
        to_connection("emb", "conv3"),
        to_connection("emb", "conv4"),
        to_connection("emb", "conv5"),
        
        to_connection("pool3", "concat"),
        to_connection("pool4", "concat"),
        to_connection("pool5", "concat"),

        # === 4. fully connected & output ===
        to_SoftMax("soft", 2, "(3,0,0)", "(concat-east)", caption="FC & Softmax"),
        to_connection("concat", "soft"),
        
        to_end()
    ]

    to_generate(arch, "textcnn_arch.tex")

if __name__ == '__main__':
    main()