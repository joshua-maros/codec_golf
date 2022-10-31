#!/usr/bin/env python3

import math
import struct
import itertools

bitstream = open('/home/josh/Documents/Sine.ogg', 'rb').read()
position = 0

def read_bit():
    global position
    byte = bitstream[position >> 3]
    assert byte != -1
    bit = (byte & (1 << (position & (8 - 1)))) > 0
    position += 1
    return bit

def read(num_bits):
    value = 0
    for bit in range(num_bits):
        value = value | ((1 if read_bit() else 0) << bit)
    return value

def read_float():
    bits = read(32)
    return struct.unpack('f', struct.pack('I', bits))[0]

def skip_to_byte_boundary():
    global position
    position = int(math.ceil(position / 8) * 8)

def assert_packet_end():
    global position
    skip_to_byte_boundary()
    assert bitstream[position // 8] == -1
    position += 8

def ilog(x):
    if x == 0:
        return 0
    else:
        return int(math.floor(math.log2(x))) + 1

oggs = ord('O') | (ord('g') << 8) | (ord('g') << 16) | (ord('S') << 24)

class OggPacket:
    def __init__(self, continuation, segments):
        self.continuation = continuation
        self.segments = segments

def read_ogg_packet():
    assert read(32) == oggs
    assert read(8) == 0 # version
    typ = read(8)
    continuation = typ & 0x1 > 0
    beginning = typ & 0x2 > 0
    end = typ & 0x4 > 0
    position = read(64)
    serial = read(32)
    page = read(32)
    checksum = read(32)
    num_segments = read(8)
    segment_lengths = []
    for _ in range(num_segments):
        segment_lengths.append(read(8))
    segments = [[]]
    for length in segment_lengths:
        data = [read(8) for _ in range(length)]
        segments[-1].extend(data)
        if length != 255:
            segments.append([])
    segments = segments[:-1]
    return OggPacket(continuation, segments)

def read_vorbis_packets():
    global position
    segments = []
    while position < len(bitstream) * 8:
        packet =read_ogg_packet()
        if packet.continuation:
            segments[-1].extend(packet.segments[0])
            packet.segments = packet.segments[1:]
        segments.extend(packet.segments)
    return [s + [-1] for s in segments]

packets = read_vorbis_packets()
bitstream = []
for packet in packets:
    bitstream.extend(packet)
position = 0

vorbis = ord('v') | (ord('o') << 8) | (ord('r') << 16) | (ord('b') << 24) | (ord('i') << 32) | (ord('s') << 40)

assert read(8) == 1 # packet type
assert read(8 * 6) == vorbis
assert read(32) == 0 # version
assert read(8) == 1 # channels
assert read(32) == 44100 # sample rate
bmax = read(32)
bavg = read(32)
bmin = read(32)
bs0 = read(4)
bs1 = read(4)
assert read(1) == 1 # framing
assert_packet_end()

assert read(8) == 3 # packet type
assert read(8 * 6) == vorbis
length = read(32)
vendor = [read(8) for _ in range(length)]
assert read(32) == 0 # num user comments
assert read(1) == 1 # framing
assert_packet_end()

class TreeNode:
    def __init__(self):
        self.value = None
        self.children = None
    
    def insert_child(self, depth, value):
        if self.value is not None:
            return False
        if depth == 0:
            if self.children is not None:
                return False
            else:
                self.value = value
                return True
        else:
            if self.children is None:
                self.children = (TreeNode(), TreeNode())
            if self.children[0].insert_child(depth - 1, value):
                return True
            else:
                return self.children[1].insert_child(depth - 1, value)
        assert False
    
    def is_full(self):
        if self.value is not None:
            return True
        elif self.children is not None:
            return self.children[0].is_full() and self.children[1].is_full()
        else:
            return False
    
    def num_leaves(self):
        if self.value is not None:
            return 1
        elif self.children is not None:
            return self.children[0].num_leaves() + self.children[1].num_leaves()
        else:
            # lol very graceful
            return -999999999999999999
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        if self.value is not None:
            return str(self.value)
        elif self.children is not None:
            return str(self.children)
        else:
            return 'Empty Tree'

assert read(8) == 5 # packet type
assert read(8 * 6) == vorbis
codebook_count = read(8) + 1
codebook_configs = []
for index in range(codebook_count):
    assert read(24) == 0x564342
    dims = read(16)
    num_entries = read(24)
    ordered = read(1) > 0
    sparse = read(1) > 0
    assert not ordered
    lengths = []
    for _ in range(num_entries):
        if sparse:
            if read(1) == 0:
                lengths.append(-1)
                continue
        lengths.append(read(5) + 1)
    tree = TreeNode()
    index = -1
    for length in lengths:
        index += 1
        if length == -1:
            continue
        tree.insert_child(length, index)
    assert tree.is_full()

    lookup_type = read(4)
    vectors = None
    if lookup_type == 0:
        pass
    elif lookup_type == 1:
        min_val = read_float()
        delta_val = read_float()
        value_bits = read(4) + 1
        sequence_p = read(1) > 0
        lookup_values = 0
        for possible_value in range(1000):
            if possible_value ** dims > num_entries:
                lookup_values = possible_value - 1
                break
        multiplicands = [read(value_bits) for _ in range(lookup_values)]

        vectors = [[0.0 for _ in range(dims)] for _ in range(num_entries)]
        for lookup_offset in range(num_entries):
            last = 0
            index_divisor = 1
            for i in range(dims):
                offset = (lookup_offset // index_divisor) % lookup_values
                val = multiplicands[offset] * delta_val + min_val + last
                vectors[lookup_offset][i]  = val
                if sequence_p:
                    last = val
                index_divisor *= lookup_values
    else:
        print('Unrecognized lookup type ' + str(lookup_type))
        assert False
    
    codebook_configs.append((tree, vectors, dims))

assert read(6) == 0 # time domain transforms
assert read(16) == 0

def read_codebook_scalar(codebook_index):
    codebook = codebook_configs[codebook_index]
    node = codebook[0]
    while node.value is None:
        node = node.children[read(1)]
    return node.value

def read_codebook_vector(codebook_index):
    index = read_codebook_scalar(codebook_index)
    return codebook_configs[codebook_index][1][index]

def low_neighbor(list, threshold_index):
    candidate = 0
    for index in range(1, threshold_index):
        if (list[index] > list[candidate] or list[candidate] < list[threshold_index]) and list[index] < list[threshold_index]:
            candidate = index
    return candidate

def high_neighbor(list, threshold_index):
    candidate = 0
    for index in range(1, threshold_index):
        if (list[index] < list[candidate] or list[candidate] < list[threshold_index]) and list[index] > list[threshold_index]:
            candidate = index
    return candidate

num_floors = read(6) + 1
floors = []
for _ in range(num_floors):
    floor_type = read(16)
    assert floor_type <= 1
    if floor_type == 0:
        assert False
    elif floor_type == 1:
        num_partitions = read(5)
        partition_classes = [read(4) for _ in range(num_partitions)]
        max_class = max(partition_classes)
        class_descriptors = []
        for _ in range(max_class + 1):
            dimension = read(3) + 1
            subclass = read(2)
            masterbook = 0
            if subclass > 0:
                masterbook = read(8)
            subclass_book = [read(8) - 1 for _ in range(2 ** subclass)]
            class_descriptors.append({ 'dimension': dimension, 'subclass': subclass, 'masterbook': masterbook, 'subclass_book': subclass_book })
        multiplier = read(2) + 1
        range_bits = read(4)
        xs = [0, 2 ** range_bits]
        for partition in range(num_partitions):
            partition_class = partition_classes[partition]
            dim = class_descriptors[partition_class]['dimension']
            for _ in range(dim):
                xs.append(read(range_bits))
        assert len(set(xs)) == len(xs)
        floors.append({'xs': xs, 'multiplier': multiplier, 'num_partitions': num_partitions, 'range_bits': range_bits, 'partition_classes': partition_classes, 'class_descriptors': class_descriptors})

num_residues = read(6) + 1
residues = []
for _ in range(num_residues):
    residue_type = read(16)
    assert residue_type <= 2
    begin = read(24)
    end = read(24)
    partition_size = read(24) + 1
    num_classifications = read(6) + 1
    classbook = read(8)
    assert num_classifications ** codebook_configs[classbook][2] <= codebook_configs[classbook][0].num_leaves()
    classifications = []
    for _ in range(num_classifications):
        classifications.append(read(3))
        if read_bit():
            classifications[-1] += read(5) * 8;
    residue_books = []
    for classification_index in range(num_classifications):
        book_set = []
        for bit in range(8):
            if (classifications[classification_index] & (1 << bit)) > 0:
                book = read(8)
                assert book < len(codebook_configs)
                assert codebook_configs[book][1] is not None
                book_set.append(book)
            else:
                book_set.append(-1)
        residue_books.append(book_set)
    residues.append({ 'type': residue_type, 'begin': begin, 'end': end, 'books': residue_books, 'classbook': classbook, 'partition_size': partition_size, 'num_classifications': num_classifications })

num_mappings = read(6) + 1
mappings = []
for _ in range(num_mappings):
    assert read(16) == 0
    num_submaps = 1
    if read_bit():
        num_submaps = read(4) + 1
    coupling_steps = 0
    if read_bit():
        assert False, 'unimplemented'
    assert read(2) == 0
    if num_submaps > 1:
        assert False, 'unimplemented'
    submaps = []
    for _ in range(num_submaps):
        read(8)
        floor = read(8)
        assert floor < len(floors)
        residue = read(8)
        assert residue < len(residues)
        submaps.append({'floor': floor, 'residue': residue})
    mappings.append({'submaps': submaps, 'coupling_steps': coupling_steps})

num_modes = read(6) + 1
modes = []
for _ in range(num_modes):
    block_flag = read_bit()
    assert read(16) == 0
    assert read(16) == 0
    mapping = read(8)
    assert mapping < len(mappings)
    modes.append({'block_flag': block_flag, 'mapping': mapping})

assert read_bit()
assert_packet_end()

while position < len(bitstream) * 8:
    assert not read_bit(), 'Expected an audio packet'
    mode_index = read(ilog(len(modes) - 1))
    mode = modes[mode_index]
    block_size = bs1 if mode['block_flag'] else bs0
    long_window = mode['block_flag']
    if long_window:
        prev_window = read_bit()
        next_window = read_bit()
        assert False, 'unimplemented'
    mapping = mappings[mode['mapping']]
    floors_now = []
    no_residue = []
    for channel_index in range(1):
        assert len(mapping['submaps']) == 1
        submap = mapping['submaps'][0]
        floor_config = floors[submap['floor']]
        nonzero = read_bit()
        no_residue.append(not nonzero)
        if not nonzero:
            floors_now.append(-1)
        else:
            floor_range = [256, 128, 86, 64][floor_config['multiplier'] - 1]
            range_bits = int(math.log2(floor_range))
            ys = []
            offset = 0
            for partition in range(floor_config['num_partitions']):
                partition_class = floor_config['class_descriptors'][floor_config['partition_classes'][partition]]
                cbits = partition_class['subclass']
                csub = 2 ** cbits - 1
                cval = 0
                if cbits > 0:
                    cval = read_codebook_scalar(partition_class['masterbook'])
                for i in range(partition_class['dimension']):
                    book = partition_class['subclass_book'][cval & csub]
                    cval = cval >> cbits
                    if book >= 0:
                        ys.append(read_codebook_scalar(book))
                    else:
                        ys.append(0)
                offset += partition_class['dimension']
            xs = floor_config['xs']
            final_ys = [ys[0], ys[1]]
            for i in range(2, len(ys)):
                low_index = low_neighbor(xs, i)
                high_index = high_neighbor(xs, i)
                slope = (final_ys[high_index] - final_ys[low_index]) / (xs[high_index] - xs[low_index])
                prediction = int(round((xs[i] - xs[low_index]) * slope + final_ys[low_index]))
                val = ys[i]
                room = 2 * min(prediction, floor_range - prediction)
                if val >= room:
                    if prediction > floor_range - prediction:
                        final_ys.append(val)
                    else:
                        final_ys.append(floor_range - val - 1)
                else:
                    final_ys.append(prediction + int(round(val / 2)))

            segments = sorted(list(zip(xs, final_ys)), key=lambda e: e[0])
            rendered_line = []
            for segment in itertools.pairwise(segments):
                x0, y0 = segment[0]
                x1, y1 = segment[1]
                for x in range(x0, x1):
                    rendered_line.append(int((y1 - y0) * (x - x0) / (x1 - x0) + y0))
            rendered_line.append(segments[-1][1])
            lut = [1.0649863e-07, 1.1341951e-07, 1.2079015e-07, 1.2863978e-07, 1.3699951e-07, 1.4590251e-07, 1.5538408e-07, 1.6548181e-07, 1.7623575e-07, 1.8768855e-07, 1.9988561e-07, 2.1287530e-07, 2.2670913e-07, 2.4144197e-07, 2.5713223e-07, 2.7384213e-07, 2.9163793e-07, 3.1059021e-07, 3.3077411e-07, 3.5226968e-07, 3.7516214e-07, 3.9954229e-07, 4.2550680e-07, 4.5315863e-07, 4.8260743e-07, 5.1396998e-07, 5.4737065e-07, 5.8294187e-07, 6.2082472e-07, 6.6116941e-07, 7.0413592e-07, 7.4989464e-07, 7.9862701e-07, 8.5052630e-07, 9.0579828e-07, 9.6466216e-07, 1.0273513e-06, 1.0941144e-06, 1.1652161e-06, 1.2409384e-06, 1.3215816e-06, 1.4074654e-06, 1.4989305e-06, 1.5963394e-06, 1.7000785e-06, 1.8105592e-06, 1.9282195e-06, 2.0535261e-06, 2.1869758e-06, 2.3290978e-06, 2.4804557e-06, 2.6416497e-06, 2.8133190e-06, 2.9961443e-06, 3.1908506e-06, 3.3982101e-06, 3.6190449e-06, 3.8542308e-06, 4.1047004e-06, 4.3714470e-06, 4.6555282e-06, 4.9580707e-06, 5.2802740e-06, 5.6234160e-06, 5.9888572e-06, 6.3780469e-06, 6.7925283e-06, 7.2339451e-06, 7.7040476e-06, 8.2047000e-06, 8.7378876e-06, 9.3057248e-06, 9.9104632e-06, 1.0554501e-05, 1.1240392e-05, 1.1970856e-05, 1.2748789e-05, 1.3577278e-05, 1.4459606e-05, 1.5399272e-05, 1.6400004e-05, 1.7465768e-05, 1.8600792e-05, 1.9809576e-05, 2.1096914e-05, 2.2467911e-05, 2.3928002e-05, 2.5482978e-05, 2.7139006e-05, 2.8902651e-05, 3.0780908e-05, 3.2781225e-05, 3.4911534e-05, 3.7180282e-05, 3.9596466e-05, 4.2169667e-05, 4.4910090e-05, 4.7828601e-05, 5.0936773e-05, 5.4246931e-05, 5.7772202e-05, 6.1526565e-05, 6.5524908e-05, 6.9783085e-05, 7.4317983e-05, 7.9147585e-05, 8.4291040e-05, 8.9768747e-05, 9.5602426e-05, 0.00010181521, 0.00010843174, 0.00011547824, 0.00012298267, 0.00013097477, 0.00013948625, 0.00014855085, 0.00015820453, 0.00016848555, 0.00017943469, 0.00019109536, 0.00020351382, 0.00021673929, 0.00023082423, 0.00024582449, 0.00026179955, 0.00027881276, 0.00029693158, 0.00031622787, 0.00033677814, 0.00035866388, 0.00038197188, 0.00040679456, 0.00043323036, 0.00046138411, 0.00049136745, 0.00052329927, 0.00055730621, 0.00059352311, 0.00063209358, 0.00067317058, 0.00071691700, 0.00076350630, 0.00081312324, 0.00086596457, 0.00092223983, 0.00098217216, 0.0010459992,  0.0011139742, 0.0011863665,  0.0012634633,  0.0013455702,  0.0014330129, 0.0015261382,  0.0016253153,  0.0017309374,  0.0018434235, 0.0019632195,  0.0020908006,  0.0022266726,  0.0023713743, 0.0025254795,  0.0026895994,  0.0028643847,  0.0030505286, 0.0032487691,  0.0034598925,  0.0036847358,  0.0039241906, 0.0041792066,  0.0044507950,  0.0047400328,  0.0050480668, 0.0053761186,  0.0057254891,  0.0060975636,  0.0064938176, 0.0069158225,  0.0073652516,  0.0078438871,  0.0083536271, 0.0088964928,  0.009474637,   0.010090352,   0.010746080, 0.011444421,   0.012188144,   0.012980198,   0.013823725, 0.014722068,   0.015678791,   0.016697687,   0.017782797, 0.018938423,   0.020169149,   0.021479854,   0.022875735, 0.024362330,   0.025945531,   0.027631618,   0.029427276, 0.031339626,   0.033376252,   0.035545228,   0.037855157, 0.040315199,   0.042935108,   0.045725273,   0.048696758, 0.051861348,   0.055231591,   0.058820850,   0.062643361, 0.066714279,   0.071049749,   0.075666962,   0.080584227, 0.085821044,   0.091398179,   0.097337747,   0.10366330, 0.11039993,    0.11757434,    0.12521498,    0.13335215, 0.14201813,    0.15124727,    0.16107617,    0.17154380, 0.18269168,    0.19456402,    0.20720788,    0.22067342, 0.23501402,    0.25028656,    0.26655159,    0.28387361, 0.30232132,    0.32196786,    0.34289114,    0.36517414, 0.38890521,    0.41417847,    0.44109412,    0.46975890, 0.50028648,    0.53279791,    0.56742212,    0.60429640, 0.64356699,    0.68538959,    0.72993007,    0.77736504, 0.82788260,    0.88168307,    0.9389798,     ]
            rendered_line = [lut[i] for i in rendered_line]
            floors_now.append(rendered_line)
    print(floors_now)
    
    if mapping['coupling_steps'] != 0:
        assert False, 'unimplemented'
    
    do_not_decode = [False for _ in range(1)]
    for submap_index in range(len(mapping['submaps'])):
        submap = mapping['submaps'][submap_index]
        ch = 0
        for channel_index in range(1):
            assert len(mapping['submaps']) == 1
            assert ch == 0
            do_not_decode[ch] = no_residue[channel_index]
            ch += 1
        residue_config = residues[submap['residue']]
        classwords_per_codeword = codebook_configs[residue_config['classbook']][2]
        residue_classifications = residue_config['num_classifications']
        residue_books = residue_config['books']
        num_to_read = residue_config['end'] - residue_config['begin']
        partitions_to_read = num_to_read // residue_config['partition_size']
        result = [[0.0 for _ in range(residue_config['partition_size'])] for _ in range(ch)]
        for pass_index in range(8):
            num_partitions = 0
            while num_partitions < partitions_to_read:
                if pass_index == 0:
                    classifications = [[0 for _ in range(partitions_to_read)] for _ in range(1)]
                    for channel_index in range(1):
                        if do_not_decode[channel_index]:
                            continue
                        codeword = read_codebook_scalar(residue_config['classbook'])
                        for classword_index in reversed(range(classwords_per_codeword)):
                            classword = codeword % residue_classifications
                            print(classword, classword_index)
                            classifications[channel_index][classword_index + num_partitions] = classword
                            codeword = codeword // residue_classifications
                for classword_index in reversed(range(classwords_per_codeword)):
                    if num_partitions >= partitions_to_read:
                        break
                    for channel_index in range(1):
                        if do_not_decode[channel_index]:
                            pass
                        classification = classifications[channel_index][num_partitions]
                        book = residue_books[classification][pass_index]
                        if book == -1:
                            continue
                        assert residue_config['type'] == 1, 'unimplemented'
                        for i in range(0, residue_config['partition_size'], classwords_per_codeword):
                            entry_temp = read_codebook_vector(book)
                            for dim in range(classwords_per_codeword):
                                result[channel_index][i + dim] += entry_temp[dim]
                    num_partitions += 1

    print(bitstream[position // 8:].index(-1), 'bytes until end')
    assert_packet_end()




