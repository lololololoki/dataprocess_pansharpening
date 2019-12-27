# -*- coding: utf-8 -*-
from genTif8bit import gen_pan, gen_mul

def main():
    srcFile = '../data/山东省21/山东省21/山东省21.tif'
    dstFile = 'test.tif'
    gen_pan(srcFile, dstFile)

if __name__ == "__main__":
    main()