import collections.abc

Sub_Head = "<H>"
No_Head = "<N>"
Htype = 1
Ntype = 0


class TreebankNode(object):
    pass


class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        self.label = label
        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)
        self.father = self.children[0].father
        self.type = self.children[0].type
        self.head = self.children[0].head
        self.left = self.children[0].left
        self.right = self.children[-1].right
        self.cun = 0
        flag = 0
        for child in self.children:
            if child.father < self.left + 1 or child.father > self.right:
                self.father = child.father
                self.type = child.type
                self.head = child.head
                flag = 1

        for child in self.children:
            if child.head != self.head:
                if child.father != self.head:
                    self.cun += 1

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0, nocache=False):
        tree = self
        sublabels = [self.label]

        while len(tree.children) == 1 and isinstance(
                tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        pre_children = []
        children = []
        sub_father = set()
        sub_head = set()
        al_make = set()

        for child in tree.children:
            sub_head |= set([child.head])
            sub_father |= set([child.father])

        for child in tree.children:
            # not in sub tree
            if (child.father in sub_head and child.father != self.head) or (
                    child.head in sub_father and child.head != self.head):
                sub_r = child.father
                if child.head in sub_father:
                    sub_r = child.head
                if sub_r not in al_make:
                    al_make |= set([sub_r])
                else:
                    continue
                sub_children = []
                for sub_child in tree.children:
                    if sub_child.father == sub_r or sub_child.head == sub_r:
                        if len(sub_children) > 0:
                            assert sub_children[-1].right == sub_child.left  # contiune span
                        sub_children.append(sub_child.convert(index=index))
                        index = sub_children[-1].right

                assert len(sub_children) > 1

                sub_node = InternalParseNode(tuple([Sub_Head]), sub_children, nocache=nocache)
                if len(children) > 0:
                    assert children[-1].right == sub_node.left  # contiune span
                children.append(sub_node)
            else:
                if len(children) > 0:
                    assert children[-1].right == child.left  # contiune span
                children.append(child.convert(index=index))
                index = children[-1].right

        return InternalParseNode(tuple(sublabels), children, nocache=nocache)


class LeafTreebankNode(TreebankNode):
    def __init__(self, tag, word, head, father, type):
        assert isinstance(tag, str)
        self.tag = tag
        self.father = father
        self.type = type
        self.head = head
        assert isinstance(word, str)
        self.word = word
        self.left = self.head - 1
        self.right = self.head

    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word, self.father, self.type)


class ParseNode(object):
    pass


class InternalParseNode(ParseNode):
    def __init__(self, label, children, nocache=False):
        assert isinstance(label, tuple)
        assert all(isinstance(sublabel, str) for sublabel in label)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right

        self.father = self.children[0].father
        self.type = self.children[0].type
        self.head = self.children[0].head
        flag = 0
        for child in self.children:
            if child.father - 1 < self.left or child.father > self.right:
                self.father = child.father
                self.type = child.type
                self.head = child.head
                flag = 1

        self.cun_w = 0
        for child in self.children:
            if self.head != child.head:
                if child.father != self.head:
                    # child.father = self.head
                    self.cun_w += 1

        self.nocache = nocache

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def chil_enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        for child in self.children:
            if child.left <= left < right <= child.right:
                return child.chil_enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()

    def oracle_type(self, left, right):
        enclosing = self.chil_enclosing(left, right)
        return enclosing.type

    def oracle_head(self, left, right):
        enclosing = self.chil_enclosing(left, right)
        return enclosing.head

    def oracle_splits(self, left, right):
        return [
            child.left
            for child in self.enclosing(left, right).children
            if left < child.left < right
        ]


class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word, father, type):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag
        self.head = index + 1
        self.father = father
        self.type = type

        assert isinstance(word, str)
        self.word = word

    def leaves(self):
        yield self

    def chil_enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        return self

    def convert(self):
        return LeafTreebankNode(self.tag, self.word, self.head, self.father, self.type)


def load_trees(path, heads=None, types=None, wordss=None, strip_top=True):
    with open(path) as infile:
        treebank = infile.read()

    return load_trees_from_str(treebank, heads, types, wordss, strip_top)


def load_trees_from_str(treebank, heads=None, types=None, wordss=None, strip_top=True):
    tokens = treebank.replace("(", " ( ").replace(")", " ) ").split()
    cun_word = 0  # without root
    cun_sent = 0

    def helper(index, flag_sent):
        nonlocal cun_sent
        nonlocal cun_word
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]

            index += 1

            if tokens[index] == "(":
                children, index = helper(index, flag_sent=0)
                if len(children) > 0:
                    tr = InternalTreebankNode(label, children)
                    trees.append(tr)
            else:
                word = tokens[index]
                index += 1
                if label != '-NONE-':
                    trees.append(LeafTreebankNode(label, word, head=cun_word + 1, father=heads[cun_sent][cun_word],
                                                  type=types[cun_sent][cun_word]))
                    if cun_sent < 0:
                        print(cun_sent, cun_word + 1, word, heads[cun_sent][cun_word], types[cun_sent][cun_word])
                    cun_word += 1

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

            if flag_sent == 1:
                cun_sent += 1
                cun_word = 0

        return trees, index

    trees, index = helper(0, flag_sent=1)
    assert index == len(tokens)
    assert len(trees) == cun_sent
    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label in ("TOP", "ROOT"):
                assert len(tree.children) == 1
                trees[i] = tree.children[0]

    def process_NONE(tree):

        if isinstance(tree, LeafTreebankNode):
            label = tree.tag
            if label == '-NONE-':
                return None
            else:
                return tree

        tr = []
        label = tree.label
        if label == '-NONE-':
            return None
        for node in tree.children:
            new_node = process_NONE(node)
            if new_node is not None:
                tr.append(new_node)
        if tr == []:
            return None
        else:
            return InternalTreebankNode(label, tr)

    new_trees = []
    for i, tree in enumerate(trees):
        new_tree = process_NONE(tree)
        new_trees.append(new_tree)
    return new_trees
