# -*- coding: utf-8 -*-
from genTif8bit import gen_pan, gen_mul

def main():
    srcFile = 'shandong21.tif'
    dstFile = 'test.tif'
    gen_pan(srcFile, dstFile)

if __name__ == "__main__":
    main()