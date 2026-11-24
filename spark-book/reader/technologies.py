"""Read beginner background for a lab exercise or find a technology by name."""
import argparse
import json
from pathlib import Path

CATALOG = Path(__file__).with_name('technology-overviews.json')


def display_topic(topic, references):
    print(f"{topic['title']}  |  Exercise {topic['first_exercise']}  |  Book p. {topic['book_page']}")
    print()
    print('\n\n'.join(topic['paragraphs']))
    if topic['references']:
        print('\nFurther reading')
        for number in topic['references']:
            ref = references[str(number)]
            print(f"[{number}] {ref['label']}. {ref['url']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('exercise', nargs='?', type=int, help='number from 1 to 251')
    mode.add_argument('--find', metavar='TEXT', help='match words in topic titles and alternate names')
    mode.add_argument('--list', action='store_true', help='list all topics alphabetically')
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text(encoding='utf-8'))
    topics = catalog['topics']
    if args.list:
        for topic in topics:
            print(f"{topic['title']}  |  Exercise {topic['first_exercise']}  |  Book p. {topic['book_page']}")
        return
    if args.find is not None:
        words = args.find.casefold().split()
        if not words:
            parser.error('enter a technology name, such as UMAP or Triton')
        matches = []
        for topic in topics:
            names = ' '.join([topic['title'], *topic['aliases']]).casefold()
            if all(word in names for word in words):
                matches.append(topic)
        if not matches:
            parser.exit(1,'No topic matched. Use --list to browse the topic names.\n')
    else:
        if not 1 <= args.exercise <= 251:
            parser.error('exercise must be from 1 to 251')
        by_slug = {topic['slug']: topic for topic in topics}
        matches = [by_slug[slug] for slug in catalog['exercises'][str(args.exercise)]]
    for index, topic in enumerate(matches):
        if index:
            print()
        display_topic(topic, catalog['references'])


if __name__ == '__main__':
    main()
