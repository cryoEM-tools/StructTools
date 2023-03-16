# functions to format contacts
import itertools
import numpy as np


convert_map_single_letter = {
    'ALA' : 'A', 'ASN' : 'N', 'CYS' : 'C', 'GLU' : 'E', 'HIS' : 'H',
    'LEU' : 'L', 'MET' : 'M', 'PRO' : 'P', 'THR' : 'T', 'TYR' : 'Y',
    'ARG' : 'R', 'ASP' : 'D', 'GLN' : 'Q', 'GLY' : 'G', 'ILE' : 'I',
    'LYS' : 'K', 'PHE' : 'F', 'SER' : 'S', 'TRP' : 'W', 'VAL' : 'V'}

convert_map_fancy = {
    'ALA' : 'Ala', 'ASN' : 'Asn', 'CYS' : 'Cys', 'GLU' : 'Glu', 'HIS' : 'His',
    'LEU' : 'Leu', 'MET' : 'Met', 'PRO' : 'Pro', 'THR' : 'Thr', 'TYR' : 'Tyr',
    'ARG' : 'Arg', 'ASP' : 'Asp', 'GLN' : 'Gln', 'GLY' : 'Gly', 'ILE' : 'Ile',
    'LYS' : 'Lys', 'PHE' : 'Phe', 'SER' : 'Ser', 'TRP' : 'Trp', 'VAL' : 'Val'}


class color:
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def _process_str_name(name):
    atom_split = name.split("-")
    res = name.split("-")[0][:3]
    resSeq = int(name.split("-")[0][3:])
    if len(atom_split) == 1:
        atom = None
    else:
        atom = atom_split[-1]
    return res, resSeq, atom


def _data_from_str(contacts_str):
    chains = []
    resSeqs = []
    resnames = []
    atoms = []
    for s in contacts_str:
        split_str = s.split(" ")      
        chains.append(int(split_str[1]))
        resnames.append(split_str[2][:3])
        split_res = split_str[2].split("-")
        if len(split_res) > 1:
            resSeqs.append(int(split_res[0][3:]))
            atoms.append(split_res[1])
        else:
            resSeqs.append(int(split_str[2][3:]))
            atoms.append(None)
    return chains, resSeqs, resnames, atoms


def _format_line(chain, res, resSeq, atom, count, style='block', indent_num=4):
    if style == 'fancy':
        resname = convert_map_fancy[res]
    elif style == 'single':
        resname = convert_map_single_letter[res]
    elif style == 'block':
        resname = res

    chainid_output = '{0: <10}'.format(str(chain))

    if atom is not None:
        resname_output = '{0: <15}'.format('%s%s-%s' % (resname, resSeq, atom))
    else:
        resname_output = '{0: <15}'.format('%s%s' % (resname, resSeq))

    count_output = '{0: <8}'.format('(%d)' % count)

    indent = "".join([" "]*indent_num)

    return "".join([indent, chainid_output, resname_output, count_output])


def _table_header(indent_num=4):
    column1 = '{0: <10}'.format('chainid')
    column2 = '{0: <15}'.format('name')
    column3 = '{0: <6}'.format('contacts')
    indent = "".join([" "]*indent_num)  
    return "".join([indent, color.UNDERLINE,column1,column2,column3,color.END])


def _output_table(chains, resnames, resSeqs, atoms, counts, style):
    output = []
    for chain, res, resSeq, atom, count in zip(
            chains, resnames, resSeqs, atoms, counts):
        output.append(
            _format_line(
                chain, res, resSeq, atom, count, style=style))
    output = "\n".join(output)
    return output


def _output_fancy(chains, resnames, resSeqs, atoms, counts, style):
    output = []
    for chain, res, resSeq, atom, count in zip(
            chains, resnames, resSeqs, atoms, counts):
        resname = convert_map_fancy[res]
        res_output = '%s%s' % (resname, resSeq)
        if atom is not None:
            res_output = '%s-%s' % (res_output, atom)
        res_output = '%s(%d)' % (res_output, count)
        output.append(res_output)
    output = ", ".join(output)
    return output


def _format_resSeqs(resSeqs, mode):
    resSeqs_sorted = np.sort(resSeqs)
    stretches = []
    start, stop, resSeqs_sorted = resSeqs_sorted[0], resSeqs_sorted[0], resSeqs_sorted[1:]
    for n in np.arange(resSeqs_sorted.shape[0]):
        if resSeqs_sorted[0] == stop + 1:
            stop, resSeqs_sorted = resSeqs_sorted[0], resSeqs_sorted[1:]
            if resSeqs_sorted.shape[0] == 0:
                stretches.append([start, stop])
        else:
            if start == stop:
                stretches.append([start])
            else:
                stretches.append([start, stop])
            start, stop, resSeqs_sorted = resSeqs_sorted[0], resSeqs_sorted[0], resSeqs_sorted[1:]
    return _stretches_to_string(stretches, mode)


def _stretches_to_string(stretches, mode):
    resSeqs_str = []
    for s in stretches:
        if len(s) == 1:
            resSeqs_str.append(str(s[0]))
        elif len(s) == 2:
            resSeqs_str.append('%s-%s' % (str(s[0]), str(s[1])))
        else:
            raise
    if mode == 'chimerax':
        resSeqs_str = ",".join(resSeqs_str)
    elif mode == 'pymol':
        resSeqs_str = "+".join(resSeqs_str)
    else:
        raise
    return resSeqs_str


def _output_pymol(chain_names, resSeqs, atoms):
    output = []
    if atoms[0] is not None:
        if chain_names is None:
            chain_names = itertools.repeat(chain_names)
        for chain, resSeq, atom in zip(chain_names, resSeqs, atoms):
            output_tmp = 'resi %d and name %s' % (resSeq, atom)
            if chain is not None:
                output_tmp = 'chain %s and %s' % (chain, output_tmp)
            output_tmp = '(%s)' % output_tmp
            output.append(output_tmp)
        output = '(%s)' % " or ".join(output)
    else:
        if chain_names is None:
            output = '(%s)' % _format_resSeqs(resSeqs, mode='pymol')
        else:
            unique_chain_names = np.unique(chain_names)
            for chain_name in unique_chain_names:
                iis = np.where(chain_names == chain_name)[0]
                resSeqs_str = _format_resSeqs(np.array(resSeqs)[iis], mode='pymol')
                output.append(
                    '(chain %s and resi %s)' % (chain_name, resSeqs_str))
            output = '(%s)' % " or ".join(output)
    return output


def _output_chimerax(chain_names, resSeqs, atoms, chimerax_model=None):
    output = []
    if atoms[0] is not None:
        if chain_names is None:
            chain_names = itertools.repeat(chain_names)
        for chain, resSeq, atom in zip(chain_names, resSeqs, atoms):
            output_tmp = ':%d@%s' % (resSeq, atom)
            if chain is not None:
                output_tmp = '/%s%s' % (chain, output_tmp)
            if chimerax_model is not None:
                output_tmp = '#%s%s' % (chimerax_model, output_tmp)
            output.append(output_tmp)
        output = " ".join(output)
    else:
        if chain_names is None:
            resSeqs_str = _format_resSeqs(resSeqs, mode='chimerax')
            output = '%s' % resSeqs_str
        else:
            unique_chain_names = np.unique(chain_names)
            for chain_name in unique_chain_names:
                iis = np.where(chain_names == chain_name)[0]
                resSeqs_str = _format_resSeqs(np.array(resSeqs)[iis], mode='chimerax')
                output_tmp = '/%s:%s' % (chain_name, resSeqs_str)
                if chimerax_model is not None:
                    output_tmp = '#%s%s' % (chimerax_model, output_tmp)
                output.append(output_tmp)
            output = " ".join(output)
    return output


def _output(
        structA_name, chainsA, resnamesA, resSeqsA, atomsA, countsA,
        structB_name, chainsB, resnamesB, resSeqsB, atomsB, countsB,
        style='block', mode='table', chainA_names=None, chainB_names=None,
        chimerax_modelA=None, chimerax_modelB=None):
    
    titleA = '%s%s%s' % (color.BOLD, structA_name, color.END)
    titleB = '\n%s%s%s' % (color.BOLD, structB_name, color.END)
    
    if mode == 'table':
        table_header = _table_header()
        outputA = _output_table(
            chainsA, resnamesA, resSeqsA, atomsA, countsA, style)
        outputB = _output_table(
            chainsB, resnamesB, resSeqsB, atomsB, countsB, style)
        
        outputA = "\n".join([table_header, outputA])
        outputB = "\n".join([table_header, outputB])
        
    elif mode == 'fancy':
        outputA = _output_fancy(
            chainsA, resnamesA, resSeqsA, atomsA, countsA, style)
        outputB = _output_fancy(
            chainsB, resnamesB, resSeqsB, atomsB, countsB, style)
        
    elif (mode == 'pymol') or (mode == 'chimerax'):
        
        if chainA_names is None:
            chainsA_names = None
        else:
            chainsA_names = np.array(chainA_names)[chainsA]
            
        if chainB_names is None:
            chainsB_names = None
        else:
            chainsB_names = np.array(chainB_names)[chainsB]

        if mode == 'pymol':  
            outputA = _output_pymol(chainsA_names, resSeqsA, atomsA)
            outputB = _output_pymol(chainsB_names, resSeqsB, atomsB)
        elif mode == 'chimerax':
            outputA = _output_chimerax(
                chainsA_names, resSeqsA, atomsA, chimerax_modelA)
            outputB = _output_chimerax(
                chainsB_names, resSeqsB, atomsA, chimerax_modelB)
        
    output = "\n".join(
        [titleA, outputA, titleB, outputB])
    return output
