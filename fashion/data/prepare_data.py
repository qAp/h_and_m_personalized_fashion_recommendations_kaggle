
import argparse
from fashion.data import HM


def main():
   parser = argparse.ArgumentParser()
   HM.add_argparse_args(parser)

   args = parser.parse_args()

   data = HM(args)
   data.prepare_data()


if __name__ == '__main__':
    main()
