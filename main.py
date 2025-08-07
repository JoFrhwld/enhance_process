import click

@click.command()
@click.argument("message")
def main(message):
    print(message)


if __name__ == "__main__":
    main()
