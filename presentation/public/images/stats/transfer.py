import click
import os
import shutil

@click.command()
@click.option('--src', default='', prompt='src', type=str)
@click.option('--dst', default='', prompt='dst', type=str)
def main(src, dst):
    shutil.copy(src, dst)

if __name__ == '__main__':
    main()