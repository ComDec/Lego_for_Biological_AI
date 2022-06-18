import os
import pathlib
import subprocess
import re
import gzip
import numpy as np
import random

def stk_to_msa_with_target(seq_id, out_dir='./'):
    '''
    Origin: https://github.com/HarveyYan/RSSMFold

    stockholm format to a simpler MSA format
    note that we shall keep inserted columns (w.r.t. the covariance model)
    as long as those columns contain target sequence residuals, and the target
    sequence shall be the first sequence in the alignment
    '''
    with open(os.path.join(out_dir, '%s.stk' % (seq_id)), 'r') as file:
        ids, desc, id2seq = [], [], {}
        for line in file:
            line = line.rstrip()
            if line.startswith('#=GS'):
                res = re.sub('\s+', ' ', line).split(' ')
                ids.append(res[1])
                desc.append(' '.join(res[3:]))

            if len(line) > 0 and not line.startswith('#') and not line.startswith('//'):
                cur_id, seq = re.sub('\s+', ' ', line).split(' ')
                if cur_id not in id2seq:
                    id2seq[cur_id] = seq
                else:
                    id2seq[cur_id] += seq

        seqs = [id2seq[cur_id] for cur_id in ids]
        # remove redundant sequences
        _, idx = np.unique(seqs, return_index=True)
        seqs = np.array(seqs)[np.sort(idx)]
    # '.' indicates insertion with regard to the covariance model
    # delete inserted columns unless they contain target nucleotides
    idx = np.where(np.array(list(seqs[0])) != '.')

    with open(os.path.join(out_dir, '%s.a2m' % (seq_id)), 'w') as file:
        for id, desc, seq in zip(ids, desc, seqs):
            file.write('>%s %s\n%s\n' % (
                id, desc, ''.join(np.array(list(seq))[idx]).replace('.', '-').upper()))

    return os.path.join(out_dir, '%s.a2m' % (seq_id))

def save_rnacmap_msa(seq, seq_id, out_dir='./', cap_rnacmap_msa_depth=50000, ncores=20,
                     dl_struct=None, ret_size=False, specify_blastn_database_path=None):
    # RNAcmap but executed in python (originally with shell)
    '''
    This is a mini program for RNAcamp, a tool for MSA searching.
    By using this, the following variables and packages are required.

        Variables
            <default_blastn_database_path>: blastn database
            <specify_blastn_database_path>: blastn database path passed by function parameters
            <rnacamp_base>: path of RNAcamp codes
            <rfam_v14_path>: path of rfam_v14 program

        Packages
            blastn
            infernal

        Return
            Null but save msa file as seqID.a2m
    '''

    with open(os.path.join(out_dir, '%s.seq' % (seq_id)), 'w') as file:
        file.write('>%s\n%s\n' % (seq_id, seq))

    if specify_blastn_database_path is None:
        blastn_database_path = default_blastn_database_path
    else:
        blastn_database_path = specify_blastn_database_path

    if not os.path.exists(blastn_database_path):
        raise ValueError(f'{blastn_database_path} does not exist')

    # first round blastn search and reformat its output
    # GC RF determines which are insertions relative to the consensus
    cmd = '''
        blastn -db {0} -query {2}.seq -out {2}.bla -evalue 0.001 -num_descriptions 1 -num_threads {3} -line_length 1000 -num_alignments {4}
        {1}/parse_blastn_local.pl {2}.bla {2}.seq {2}.aln
        {1}/reformat.pl fas sto {2}.aln {2}.sto
        '''.format(blastn_database_path, rnacmap_base, os.path.join(out_dir, seq_id), ncores, cap_rnacmap_msa_depth)
    subprocess.call(cmd, shell=True)

    if dl_struct is None:
        # RNAfold for only the target sequence
        cmd = '''
        RNAfold {0}.seq --noPS | awk '{{print $1}}' | tail -n +3 > {0}.dbn
        for i in `awk '{{print $2}}' {0}.sto | head -n5 | tail -n1 | grep -b -o - | sed 's/..$//'`; do sed -i "s/./&-/$i" {0}.dbn; done
        head -n -1 {0}.sto > {1}.sto
        echo "#=GC SS_cons                     "`cat {0}.dbn` > {1}.txt
        cat {1}.sto {1}.txt > {0}.sto
        echo "//" >> {0}.sto
        '''.format(os.path.join(out_dir, seq_id), os.path.join(out_dir, 'temp'))
        subprocess.call(cmd, shell=True)
    else:
        # deep learning predicted structure
        cmd = '''
        echo "{2}" > {0}.dbn
        for i in `awk '{{print $2}}' {0}.sto | head -n5 | tail -n1 | grep -b -o - | sed 's/..$//'`; do sed -i "s/./&-/$i" {0}.dbn; done
        head -n -1 {0}.sto > {1}.sto
        echo "#=GC SS_cons                     "`cat {0}.dbn` > {1}.txt
        cat {1}.sto {1}.txt > {0}.sto
        echo "//" >> {0}.sto
        '''.format(os.path.join(out_dir, seq_id), os.path.join(out_dir, 'temp'), dl_struct)
        subprocess.call(cmd, shell=True)

    # second round covariance model search
    cmd = '''
    cmbuild --hand -F {0}.cm {0}.sto >/dev/null 2>&1
	cmcalibrate {0}.cm >/dev/null 2>&1
	cmsearch -o {0}.out -A {0}.msa --cpu {1} --incE 10.0 {0}.cm {2} >/dev/null 2>&1
	{3} --replace acgturyswkmbdhvn:................ a2m {0}.msa > {4}.a2m
    '''.format(os.path.join(out_dir, seq_id), ncores, blastn_database_path, os.path.join(rfam_v14_path, 'esl-reformat'),
               os.path.join(out_dir, 'temp'))
    ret_code = subprocess.call(cmd, shell=True)

    if ret_code == 0:
        # constrain maximal depth of the MSA
        with open(os.path.join(out_dir, 'temp.a2m'), 'r') as in_file:
            all_id, all_seq = [], []
            seq_ = ''
            for line in in_file:
                if line.startswith('>'):
                    all_id.append(line.rstrip())
                    if len(seq_) > 0:
                        all_seq.append(seq_)
                        seq_ = ''
                elif len(line) > 0:
                    seq_ += line.rstrip().replace('T', 'U').upper()
            if len(seq_) > 0:
                all_seq.append(seq_)

            if len(all_seq) > cap_rnacmap_msa_depth:
                selected_idx = np.sort(np.random.choice(np.arange(len(all_seq)), cap_rnacmap_msa_depth, False))
                all_id = np.array(all_id)[selected_idx]
                all_seq = np.array(all_seq)[selected_idx]
    else:
        # when <seq_id>.msa is empty
        all_id, all_seq = [], []

    with open(os.path.join(out_dir, '%s.a2m' % (seq_id)), 'w') as out_file:
        out_file.write('>target_seq %s\n%s\n' % (seq_id, seq.upper()))  # imperative to call an `upper' here
        for id_, seq_ in zip(all_id, all_seq):
            if len(seq_) <= 2000:
                # need to restrict sequence length, vis-a-vis RFAM 02541
                out_file.write('%s\n%s\n' % (id_, seq_))

    if ret_size:
        return os.path.join(out_dir, '%s.a2m' % (seq_id)), len(all_seq)
    else:
        return os.path.join(out_dir, '%s.a2m' % (seq_id))


def filter_gaps(msa, gap_cutoff=0.5):
    '''
    filter alignment to remove gappy positions
    '''
    VOCAB = ['A', 'C', 'G', 'U', '-']
    N_VOCAB = len(VOCAB)
    frac_gaps = np.mean((msa == N_VOCAB - 1).astype(np.float), 0)
    # mostly non-gaps, or containing target residuals
    col_idx = np.where((frac_gaps < gap_cutoff) | (msa[0] != (N_VOCAB - 1)))[0]
    return msa[:, col_idx], col_idx

'''
The following two utils are depented.
jit and prange origins from numba
'''

@jit(nopython=True, parallel=True)
def get_msa_eff(msa, eff_cutoff):
    # care not to trigger the race condition — multiple threads trying to write
    # the same slice/element in an array simultaneously
    msa_depth, msa_length = msa.shape[0], msa.shape[1]
    seq_weights = np.zeros(msa_depth)
    for i in prange(msa_depth):
        seq_i = msa[i]
        for j in range(msa_depth):
            seq_j = msa[j]
            if np.sum(seq_i == seq_j) / msa_length >= eff_cutoff:
                seq_weights[i] += 1
    seq_weights = 1 / seq_weights
    return seq_weights

def read_msa(msa_filepath, cap_msa_depth=np.inf, eff_cutoff=0.8, gap_cutoff=0.5):
    '''
    read and preprocess MSA
    - set non-vocab characters to gaps
    - char to int
    - cap MSA depth by sampling
    - filter gaps
    - get sequence weights
    '''
    VOCAB = ['A', 'C', 'G', 'U', '-']
    N_VOCAB = len(VOCAB)
    msa = []
    with open(msa_filepath, 'r') as file:
        for line in file:
            if not line.startswith('>'):
                # non-vocab characters to gaps
                msa.append([char if char in VOCAB else VOCAB[-1] for char in line.rstrip()])
    msa = np.array(msa)

    # char to int
    for n in range(len(VOCAB)):
        msa[msa == VOCAB[n]] = n
    msa = msa.astype(np.int)

    # 1. cap MSA depth if necessary
    # 2. remove gaps if necessary
    if msa.shape[0] > cap_msa_depth:
        # for weighted sampling — more unique sequences
        msa_w = get_msa_eff(msa[1:], eff_cutoff)
        sampled_idx = np.random.choice(np.arange(1, msa.shape[0]), cap_msa_depth - 1, False, msa_w / np.sum(msa_w))
        sampled_idx = np.sort(sampled_idx)
        msa = np.concatenate([msa[:1, :], msa[sampled_idx]], axis=0)

    msa, v_idx = filter_gaps(msa, gap_cutoff)
    msa_w = get_msa_eff(msa, eff_cutoff)

    return msa, msa_w, v_idx

NUC_VOCAB = ['A', 'C', 'G', 'U', 'N']
basedir = pathlib.Path(RSSMFold.__file__).parent.parent.resolve()
linearpartition_executable = os.path.join(basedir, 'LinearPartition', 'linearpartition')


def read_fasta_file(fasta_path):
    all_ids, all_seqs = [], []
    with open(fasta_path, 'r') as file:
        read_seq = ''
        for line in file:
            line = line.rstrip()
            if line[0] == '>':
                seq_id = line[1:].rstrip().lstrip()
                all_ids.append(seq_id)
                if len(read_seq) > 0:
                    all_seqs.append(read_seq)
                    read_seq = ''
            else:
                seq_one_line = line.upper().replace('T', 'U')
                seq_one_line = ''.join(list(map(lambda c: c if c in NUC_VOCAB else 'N', seq_one_line)))
                read_seq += seq_one_line
        if len(read_seq) > 0:
            all_seqs.append(read_seq)
    return all_ids, all_seqs


def augment_linearpartition(seqs, cutoff, outdir):
    '''
    This function requires LinearPartition program, which can be downloaded and installed with:

        git clone https://github.com/LinearFold/LinearPartition
        cd LinearPartition
        make
        cd ..

    after that, create new variable:
        <linearpartition_executable>: path of linearpartition


    '''
    all_triu = []
    for seq in seqs:
        np.random.seed(random.seed())
        outfile = os.path.join(outdir, str(np.random.rand()) + '.lp_out')
        cmd = f'echo {seq} | {linearpartition_executable} -o {outfile} -c {cutoff} >/dev/null 2>&1'
        subprocess.call(cmd, shell=True)
        nb_nodes = len(seq)
        pred_mat = np.zeros((nb_nodes, nb_nodes, 1), dtype=np.float32)
        with open(outfile, 'r') as file:
            for line in file:
                ret = line.rstrip().split()
                if len(ret) == 0:
                    continue
                row = int(ret[0]) - 1
                col = int(ret[1]) - 1
                prob = float(ret[2])
                pred_mat[row, col] = prob
        all_triu.append(pred_mat[np.triu_indices(nb_nodes)])
        os.remove(outfile)
    return np.concatenate(all_triu, axis=0)


def bpseq_remove_pseudoknots(bpseq_path):
    '''
    Make sure you have source code of FreeKnot.
    Note the bash command below, if your program install in other place, please mix the following command.
    '''

    subprocess.call('''export PERLLIB=./FreeKnot
        perl FreeKnot/remove_pseudoknot.pl -i bpseq -s bp {0} > {0}_freeknot'''.format(bpseq_path), shell=True)


def bpseq_to_dot_bracket(bpseq_path):
    '''
    convert bpseq file(with csv format, it's clever?) into dot-bracket string.

    Requirement: import pandas as pd
    '''

    bpseq_remove_pseudoknots(bpseq_path)
    file = pd.read_csv(bpseq_path + '_freeknot', delimiter=' ', header=None)
    seq = ''.join(list(file.iloc[:, 1]))
    struct = ['.'] * len(seq)
    for i, idx in enumerate(list(file.iloc[:, 2])):
        if idx != 0:
            if i < idx - 1:
                struct[i] = '('
            else:
                struct[i] = ')'
    return ''.join(struct)

