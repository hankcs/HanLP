# -*- coding:utf-8 -*-
# Modified from https://github.com/tylerneylon/explacy
import io
from collections import defaultdict
from pprint import pprint

from phrasetree.tree import Tree


def make_table(rows, insert_header=False):
    col_widths = [max(len(s) for s in col) for col in zip(*rows[1:])]
    rows[0] = [x[:l] for x, l in zip(rows[0], col_widths)]
    fmt = '\t'.join('%%-%ds' % width for width in col_widths)
    if insert_header:
        rows.insert(1, ['─' * width for width in col_widths])
    return '\n'.join(fmt % tuple(row) for row in rows)


def _start_end(arrow):
    start, end = arrow['from'], arrow['to']
    mn = min(start, end)
    mx = max(start, end)
    return start, end, mn, mx


def pretty_tree_horizontal(arrows, _do_print_debug_info=False):
    """Print the dependency tree horizontally

    Args:
      arrows: 
      _do_print_debug_info:  (Default value = False)

    Returns:

    """
    # Set the base height; these may increase to allow room for arrowheads after this.
    arrows_with_deps = defaultdict(set)
    for i, arrow in enumerate(arrows):
        arrow['underset'] = set()
        if _do_print_debug_info:
            print('Arrow %d: "%s" -> "%s"' % (i, arrow['from'], arrow['to']))
        num_deps = 0
        start, end, mn, mx = _start_end(arrow)
        for j, other in enumerate(arrows):
            if arrow is other:
                continue
            o_start, o_end, o_mn, o_mx = _start_end(other)
            if ((start == o_start and mn <= o_end <= mx) or
                    (start != o_start and mn <= o_start <= mx)):
                num_deps += 1
                if _do_print_debug_info:
                    print('%d is over %d' % (i, j))
                arrow['underset'].add(j)
        arrow['num_deps_left'] = arrow['num_deps'] = num_deps
        arrows_with_deps[num_deps].add(i)

    if _do_print_debug_info:
        print('')
        print('arrows:')
        pprint(arrows)

        print('')
        print('arrows_with_deps:')
        pprint(arrows_with_deps)

    # Render the arrows in characters. Some heights will be raised to make room for arrowheads.
    sent_len = (max([max(arrow['from'], arrow['to']) for arrow in arrows]) if arrows else 0) + 1
    lines = [[] for i in range(sent_len)]
    num_arrows_left = len(arrows)
    while num_arrows_left > 0:

        assert len(arrows_with_deps[0])

        arrow_index = arrows_with_deps[0].pop()
        arrow = arrows[arrow_index]
        src, dst, mn, mx = _start_end(arrow)

        # Check the height needed.
        height = 3
        if arrow['underset']:
            height = max(arrows[i]['height'] for i in arrow['underset']) + 1
        height = max(height, 3, len(lines[dst]) + 3)
        arrow['height'] = height

        if _do_print_debug_info:
            print('')
            print('Rendering arrow %d: "%s" -> "%s"' % (arrow_index,
                                                        arrow['from'],
                                                        arrow['to']))
            print('  height = %d' % height)

        goes_up = src > dst

        # Draw the outgoing src line.
        if lines[src] and len(lines[src]) < height:
            lines[src][-1].add('w')
        while len(lines[src]) < height - 1:
            lines[src].append(set(['e', 'w']))
        if len(lines[src]) < height:
            lines[src].append({'e'})
        lines[src][height - 1].add('n' if goes_up else 's')

        # Draw the incoming dst line.
        lines[dst].append(u'►')
        while len(lines[dst]) < height:
            lines[dst].append(set(['e', 'w']))
        lines[dst][-1] = set(['e', 's']) if goes_up else set(['e', 'n'])

        # Draw the adjoining vertical line.
        for i in range(mn + 1, mx):
            while len(lines[i]) < height - 1:
                lines[i].append(' ')
            lines[i].append(set(['n', 's']))

        # Update arrows_with_deps.
        for arr_i, arr in enumerate(arrows):
            if arrow_index in arr['underset']:
                arrows_with_deps[arr['num_deps_left']].remove(arr_i)
                arr['num_deps_left'] -= 1
                arrows_with_deps[arr['num_deps_left']].add(arr_i)

        num_arrows_left -= 1

    return render_arrows(lines)


def render_arrows(lines):
    arr_chars = {'ew': u'─',
                 'ns': u'│',
                 'en': u'└',
                 'es': u'┌',
                 'enw': u'┴',
                 'ensw': u'┼',
                 'ens': u'├',
                 'esw': u'┬'}
    # Convert the character lists into strings.
    max_len = max(len(line) for line in lines)
    for i in range(len(lines)):
        lines[i] = [arr_chars[''.join(sorted(ch))] if type(ch) is set else ch for ch in lines[i]]
        lines[i] = ''.join(reversed(lines[i]))
        lines[i] = ' ' * (max_len - len(lines[i])) + lines[i]
    return lines


def render_span(begin, end, unidirectional=False):
    if end - begin == 1:
        return ['───►']
    elif end - begin == 2:
        return [
            '──┐',
            '──┴►',
        ] if unidirectional else [
            '◄─┐',
            '◄─┴►',
        ]

    rows = []
    for i in range(begin, end):
        if i == (end - begin) // 2 + begin:
            rows.append('  ├►')
        elif i == begin:
            rows.append('──┐' if unidirectional else '◄─┐')
        elif i == end - 1:
            rows.append('──┘' if unidirectional else '◄─┘')
        else:
            rows.append('  │')
    return rows


def tree_to_list(T):
    return [T.label(), [tree_to_list(t) if isinstance(t, Tree) else t for t in T]]


def list_to_tree(L):
    if isinstance(L, str):
        return L
    return Tree(L[0], [list_to_tree(child) for child in L[1]])


def render_labeled_span(b, e, spans, labels, label, offset, unidirectional=False):
    spans.extend([''] * (b - offset))
    spans.extend(render_span(b, e, unidirectional))
    center = b + (e - b) // 2
    labels.extend([''] * (center - offset))
    labels.append(label)
    labels.extend([''] * (e - center - 1))


def main():
    # arrows = [{'from': 1, 'to': 0}, {'from': 2, 'to': 1}, {'from': 2, 'to': 4}, {'from': 2, 'to': 5},
    #           {'from': 4, 'to': 3}]
    # lines = pretty_tree_horizontal(arrows)
    # print('\n'.join(lines))
    # print('\n'.join([
    #     '◄─┐',
    #     '  │',
    #     '  ├►',
    #     '  │',
    #     '◄─┘',
    # ]))
    print('\n'.join(render_span(7, 12)))


if __name__ == '__main__':
    main()
left_rule = {'<': ':', '^': ':', '>': '-'}
right_rule = {'<': '-', '^': ':', '>': ':'}


def evalute_field(record, field_spec):
    """Evalute a field of a record using the type of the field_spec as a guide.

    Args:
      record:
      field_spec:

    Returns:

    """
    if type(field_spec) is int:
        return str(record[field_spec])
    elif type(field_spec) is str:
        return str(getattr(record, field_spec))
    else:
        return str(field_spec(record))


def markdown_table(headings, records, fields=None, alignment=None, file=None):
    """Generate a Doxygen-flavor Markdown table from records.
    See https://stackoverflow.com/questions/13394140/generate-markdown-tables

    file -- Any object with a 'write' method that takes a single string
        parameter.
    records -- Iterable.  Rows will be generated from this.
    fields -- List of fields for each row.  Each entry may be an integer,
        string or a function.  If the entry is an integer, it is assumed to be
        an index of each record.  If the entry is a string, it is assumed to be
        a field of each record.  If the entry is a function, it is called with
        the record and its return value is taken as the value of the field.
    headings -- List of column headings.
    alignment - List of pairs alignment characters.  The first of the pair
        specifies the alignment of the header, (Doxygen won't respect this, but
        it might look good, the second specifies the alignment of the cells in
        the column.

        Possible alignment characters are:
            '<' = Left align
            '>' = Right align (default for cells)
            '^' = Center (default for column headings)

    Args:
      headings:
      records:
      fields:  (Default value = None)
      alignment:  (Default value = None)
      file:  (Default value = None)

    Returns:

    """
    if not file:
        file = io.StringIO()
    num_columns = len(headings)
    if not fields:
        fields = list(range(num_columns))
    assert len(headings) == num_columns

    # Compute the table cell data
    columns = [[] for i in range(num_columns)]
    for record in records:
        for i, field in enumerate(fields):
            columns[i].append(evalute_field(record, field))

    # Fill out any missing alignment characters.
    extended_align = alignment if alignment is not None else [('^', '<')]
    if len(extended_align) > num_columns:
        extended_align = extended_align[0:num_columns]
    elif len(extended_align) < num_columns:
        extended_align += [('^', '>') for i in range(num_columns - len(extended_align))]

    heading_align, cell_align = [x for x in zip(*extended_align)]

    field_widths = [len(max(column, key=len)) if len(column) > 0 else 0
                    for column in columns]
    heading_widths = [max(len(head), 2) for head in headings]
    column_widths = [max(x) for x in zip(field_widths, heading_widths)]

    _ = ' | '.join(['{:' + a + str(w) + '}'
                    for a, w in zip(heading_align, column_widths)])
    heading_template = '| ' + _ + ' |'
    _ = ' | '.join(['{:' + a + str(w) + '}'
                    for a, w in zip(cell_align, column_widths)])
    row_template = '| ' + _ + ' |'

    _ = ' | '.join([left_rule[a] + '-' * (w - 2) + right_rule[a]
                    for a, w in zip(cell_align, column_widths)])
    ruling = '| ' + _ + ' |'

    file.write(heading_template.format(*headings).rstrip() + '\n')
    file.write(ruling.rstrip() + '\n')
    for row in zip(*columns):
        file.write(row_template.format(*row).rstrip() + '\n')
    if isinstance(file, io.StringIO):
        text = file.getvalue()
        file.close()
        return text
