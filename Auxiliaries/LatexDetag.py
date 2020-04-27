"""
Small collection of functions for removing latex commands, blocks, etc.
The purpose of this is to prepare latex source code for word counting or 
grammar check in grammarly.
Tags could be removed by hand, of course, but it is a very boring routine.

Example of use:
with open("Chapter.txt", "r") as mf:
    txt = mf.read()

txt2 = PrepareForWordCounting(txt, std_del_single_tags, std_keep_tags)
with open("ChapterX.txt", "w") as mf:
    mf.write(txt2)
"""

LIM = 50 #limit of how many inner commands are allowed to be in one block

def GetTagsArea(text, tag1, tag2, i=0):    
    start_i = text.find(tag1, i)
    if start_i == -1:
        return -1,-1, 0, 0
    start_i += len(tag1)
    stop_i = text.find(tag2, start_i)
    limit_cnts = 0
    substr = text[start_i : stop_i]    
    while ( substr.count("{") > substr.count("}") ) and limit_cnts<LIM:
        stop_i = text.find(tag2, stop_i+1)
        substr = text[start_i : stop_i]
        limit_cnts += 1
    return start_i, stop_i, len(tag1), len(tag2)

def GetBlockArea(text, tag, i=0):
    return GetTagsArea(
        text,
        f"\\begin{{{tag}}}",
        f"\\end{{{tag}}}",
        i
        )
    
def GetBlockMathArea(text, i=0):
    return GetTagsArea(
        text,
        "$$",
        "$$",
        i
        )

def GetInlineMathArea(text, i=0):    
    text = text.replace('$$', '**')
    return GetTagsArea(
        text,
        "$",
        "$",
        i
        )

def GetCommentArea(text, i=0):
    IDpair = chr(200)+chr(201)
    safe_text = text.replace("\\%", IDpair)
    return GetTagsArea(
        safe_text,
        "%",
        "\n",
        i
        )

def RemoveFirstElement(text, selector, func=None):
    i_stop = 0
    i_start, i_stop, n1, n2 = selector(text, 0)
    
    if i_start == -1 or i_stop == -1:
        return text, False

    text_new = text[0 : i_start-n1]
    if func == None:
        substr = text[i_start : i_stop]
    else:
        substr = func(text[i_start : i_stop])    
    text_new += substr
    text_new += text[i_stop+n2 : ]        
    return text_new, True

def RemoveAllElements(text, selector, func):
    go = True
    while go:
        text_new, status = selector(text, func)
        go = status
        if go:
            text = text_new
    return text

def RemoveAllComments(text):
    def RemoveFunc(text, function):
        return RemoveFirstElement(text, GetCommentArea, function)
    return RemoveAllElements(text, RemoveFunc, delete)

def RemoveAllTags(text, tag, func=None):    
    def spec(text, i):
        return GetTagsArea(text, tag+"{", "}", i)
    def RemoveFunc(text, function):
        return RemoveFirstElement(text, spec, function)
    return RemoveAllElements(text, RemoveFunc, func)

def RemoveAllInlineMath(text, func=None):
    def RemoveFunc(text, function):
        return RemoveFirstElement(text, GetInlineMathArea, function)
    return RemoveAllElements(text, RemoveFunc, func)

def RemoveAllBlockMath(text, func=None):
    def RemoveFunc(text, function):
        return RemoveFirstElement(text, GetBlockMathArea, function)
    return RemoveAllElements(text, RemoveFunc, func)

def RemoveAllBlocks(text, tag, func=None):
    def spec(text, i):
        return GetBlockArea(text, tag, i)
    def RemoveFunc(text, function):
        return RemoveFirstElement(text, spec, function)
    return RemoveAllElements(text, RemoveFunc, func)



def delete (text): return ""

def KeepCaption(text):
    if 'caption' in text:
        i0, i1, n1, n2 = GetTagsArea(text, '\\caption{', '}', 0)
        caption_text = text[i0:i1]
        return caption_text
    else:
        return ''

def Replace(text, sub="<NONE>"):
    return sub

std_keep_tags = ['\\textbf', '\\emph', '\\section', '\\subsection', '\\subsubsection', '\\textsuperscript', '\\textsuperscript']
std_del_single_tags = ['\\vspace', '\\hspace', '\\\\']

def PrepareForWordCounting(text, tagstodel=[], tagstokeep=[], substr_to_del=[]):
    #delete comments
    text = RemoveAllComments(text)
    #delete tables
    text = RemoveAllBlocks(text, 'table', KeepCaption)

    #delete \label
    text = RemoveAllTags(text, '\\label', delete)
    #replace '~' with ' '
    text = text.replace("~", ' ')
    
    #keep figure caption,
    text = RemoveAllBlocks(text, 'figure', KeepCaption)
    text = RemoveAllBlocks(text, 'figure*', KeepCaption)

    #replace math with 1 word <math>
    text = RemoveAllBlocks(text, 'equation', delete)
    text = RemoveAllBlocks(text, 'equation*', delete)
    text = RemoveAllInlineMath(text, lambda x: Replace(x, 'X'))    
    text = RemoveAllBlockMath(text, lambda x: Replace(x, '<MATH>'))
    
    #replace \ref with 0
    text = RemoveAllTags(text, '\\ref', lambda x: Replace(x, '0'))

    #replace \cite with [0]
    text = RemoveAllTags(text, '\\cite', lambda x: Replace(x, '[0]'))
    
    #remove all tags to delete:
    for tag in tagstodel:
        text = RemoveAllTags(text, tag, delete)

    for tag in tagstokeep:
        text = RemoveAllTags(text, tag, None)    
        
    for sub in substr_to_del:
        text = text.replace(sub, '')

    return text

def count_words(text):
    # Cleaning text and lower casing all words
    for char in '-.,\n:()[]}{_^$$+*\t':
        text=text.replace(char,' ')
    text = ' '.join(text.split())  
    word_list = text.split()
    return len(word_list)


