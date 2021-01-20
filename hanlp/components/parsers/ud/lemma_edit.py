"""
Utilities for processing lemmas

Adopted from UDPipe Future
https://github.com/CoNLL-UD-2018/UDPipe-Future
"""


def min_edit_script(source, target, allow_copy=False):
    """Finds the minimum edit script to transform the source to the target

    Args:
      source: 
      target: 
      allow_copy:  (Default value = False)

    Returns:

    """
    a = [[(len(source) + len(target) + 1, None)] * (len(target) + 1) for _ in range(len(source) + 1)]
    for i in range(0, len(source) + 1):
        for j in range(0, len(target) + 1):
            if i == 0 and j == 0:
                a[i][j] = (0, "")
            else:
                if allow_copy and i and j and source[i - 1] == target[j - 1] and a[i - 1][j - 1][0] < a[i][j][0]:
                    a[i][j] = (a[i - 1][j - 1][0], a[i - 1][j - 1][1] + "→")
                if i and a[i - 1][j][0] < a[i][j][0]:
                    a[i][j] = (a[i - 1][j][0] + 1, a[i - 1][j][1] + "-")
                if j and a[i][j - 1][0] < a[i][j][0]:
                    a[i][j] = (a[i][j - 1][0] + 1, a[i][j - 1][1] + "+" + target[j - 1])
    return a[-1][-1][1]


def gen_lemma_rule(form, lemma, allow_copy=False):
    """Generates a lemma rule to transform the source to the target

    Args:
      form: 
      lemma: 
      allow_copy:  (Default value = False)

    Returns:

    """
    form = form.lower()

    previous_case = -1
    lemma_casing = ""
    for i, c in enumerate(lemma):
        case = "↑" if c.lower() != c else "↓"
        if case != previous_case:
            lemma_casing += "{}{}{}".format("¦" if lemma_casing else "", case,
                                            i if i <= len(lemma) // 2 else i - len(lemma))
        previous_case = case
    lemma = lemma.lower()

    best, best_form, best_lemma = 0, 0, 0
    for l in range(len(lemma)):
        for f in range(len(form)):
            cpl = 0
            while f + cpl < len(form) and l + cpl < len(lemma) and form[f + cpl] == lemma[l + cpl]: cpl += 1
            if cpl > best:
                best = cpl
                best_form = f
                best_lemma = l

    rule = lemma_casing + ";"
    if not best:
        rule += "a" + lemma
    else:
        rule += "d{}¦{}".format(
            min_edit_script(form[:best_form], lemma[:best_lemma], allow_copy),
            min_edit_script(form[best_form + best:], lemma[best_lemma + best:], allow_copy),
        )
    return rule


def apply_lemma_rule(form, lemma_rule):
    """Applies the lemma rule to the form to generate the lemma

    Args:
      form: 
      lemma_rule: 

    Returns:

    """
    cells = lemma_rule.split(";", 1)
    if len(cells) == 1:  # Some predicted lemma rules are _, which might be due to partial annotation
        return form.lower()
    casing, rule = cells
    if rule.startswith("a"):
        lemma = rule[1:]
    else:
        form = form.lower()
        rules, rule_sources = rule[1:].split("¦"), []
        assert len(rules) == 2
        for rule in rules:
            source, i = 0, 0
            while i < len(rule):
                if rule[i] == "→" or rule[i] == "-":
                    source += 1
                else:
                    assert rule[i] == "+"
                    i += 1
                i += 1
            rule_sources.append(source)

        try:
            lemma, form_offset = "", 0
            for i in range(2):
                j, offset = 0, (0 if i == 0 else len(form) - rule_sources[1])
                while j < len(rules[i]):
                    if rules[i][j] == "→":
                        lemma += form[offset]
                        offset += 1
                    elif rules[i][j] == "-":
                        offset += 1
                    else:
                        assert (rules[i][j] == "+")
                        lemma += rules[i][j + 1]
                        j += 1
                    j += 1
                if i == 0:
                    lemma += form[rule_sources[0]: len(form) - rule_sources[1]]
        except:
            lemma = form

    for rule in casing.split("¦"):
        if rule == "↓0": continue  # The lemma is lowercased initially
        case, offset = rule[0], int(rule[1:])
        lemma = lemma[:offset] + (lemma[offset:].upper() if case == "↑" else lemma[offset:].lower())

    return lemma
