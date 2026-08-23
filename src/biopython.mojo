"""C ABI kernels for pairwise alignment and sequence-file parsing."""

from std.sys.info import simd_width_of

comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime NEG = -1.0e300


def _max3(a: Float64, b: Float64, c: Float64) -> Float64:
    return max(a, max(b, c))


def _best_state(a: Float64, b: Float64, c: Float64) -> UInt8:
    if a >= b and a >= c:
        return UInt8(0)
    if b >= c:
        return UInt8(1)
    return UInt8(2)


def _alignment_score(
    target: BPtr,
    query: BPtr,
    nt: Int,
    nq: Int,
    match_score: Float64,
    mismatch_score: Float64,
    open_gap_score: Float64,
    extend_gap_score: Float64,
    local: Bool,
    work: FPtr,
) -> Float64:
    var cols = nq + 1
    var pm = work
    var pd = work.unsafe_offset(cols)
    var pi = work.unsafe_offset(2 * cols)
    var cm = work.unsafe_offset(3 * cols)
    var cd = work.unsafe_offset(4 * cols)
    var ci = work.unsafe_offset(5 * cols)

    if local:
        for j in range(cols):
            pm[unsafe_offset=j] = 0.0
            pd[unsafe_offset=j] = 0.0
            pi[unsafe_offset=j] = 0.0
    else:
        pm[unsafe_offset=0] = 0.0
        pd[unsafe_offset=0] = NEG
        pi[unsafe_offset=0] = NEG
        for j in range(1, cols):
            pm[unsafe_offset=j] = NEG
            pd[unsafe_offset=j] = NEG
            pi[unsafe_offset=j] = (
                open_gap_score + Float64(j - 1) * extend_gap_score
            )

    var best = 0.0 if local else NEG
    comptime W = simd_width_of[DType.float64]()
    for i in range(1, nt + 1):
        if local:
            cm[unsafe_offset=0] = 0.0
            cd[unsafe_offset=0] = 0.0
            ci[unsafe_offset=0] = 0.0
        else:
            cm[unsafe_offset=0] = NEG
            cd[unsafe_offset=0] = (
                open_gap_score + Float64(i - 1) * extend_gap_score
            )
            ci[unsafe_offset=0] = NEG
        var j = 1
        while j + W <= cols:
            var query_chars = query.unsafe_load[width=W](j - 1)
            var matches = query_chars.eq(
                SIMD[DType.uint8, W](target[unsafe_offset=i - 1])
            )
            var subs = matches.select(
                SIMD[DType.float64, W](match_score),
                SIMD[DType.float64, W](mismatch_score),
            )
            var pmd = pm.unsafe_load[width=W](j - 1)
            var pdd = pd.unsafe_load[width=W](j - 1)
            var pid = pi.unsafe_load[width=W](j - 1)
            var mv = max(pmd, max(pdd, pid)) + subs
            var pmu = pm.unsafe_load[width=W](j)
            var pdu = pd.unsafe_load[width=W](j)
            var piu = pi.unsafe_load[width=W](j)
            var dv = max(
                pmu + open_gap_score,
                max(pdu + extend_gap_score, piu + open_gap_score),
            )
            if local:
                mv = max(SIMD[DType.float64, W](0.0), mv)
                dv = max(SIMD[DType.float64, W](0.0), dv)
            var iv = SIMD[DType.float64, W]()
            comptime for lane in range(W):
                if lane == 0:
                    iv[lane] = _max3(
                        cm[unsafe_offset=j - 1] + open_gap_score,
                        cd[unsafe_offset=j - 1] + open_gap_score,
                        ci[unsafe_offset=j - 1] + extend_gap_score,
                    )
                else:
                    iv[lane] = _max3(
                        mv[lane - 1] + open_gap_score,
                        dv[lane - 1] + open_gap_score,
                        iv[lane - 1] + extend_gap_score,
                    )
                if local:
                    iv[lane] = max(0.0, iv[lane])
            if local:
                best = max(best, max(mv, max(dv, iv)).reduce_max())
            cm.unsafe_store(j, mv)
            cd.unsafe_store(j, dv)
            ci.unsafe_store(j, iv)
            j += W
        while j < cols:
            var sub = (
                match_score if target[unsafe_offset=i - 1]
                == query[unsafe_offset=j - 1] else mismatch_score
            )
            var mv = (
                _max3(
                    pm[unsafe_offset=j - 1],
                    pd[unsafe_offset=j - 1],
                    pi[unsafe_offset=j - 1],
                )
                + sub
            )
            var dv = _max3(
                pm[unsafe_offset=j] + open_gap_score,
                pd[unsafe_offset=j] + extend_gap_score,
                pi[unsafe_offset=j] + open_gap_score,
            )
            var iv = _max3(
                cm[unsafe_offset=j - 1] + open_gap_score,
                cd[unsafe_offset=j - 1] + open_gap_score,
                ci[unsafe_offset=j - 1] + extend_gap_score,
            )
            if local:
                mv = max(0.0, mv)
                dv = max(0.0, dv)
                iv = max(0.0, iv)
                best = max(best, _max3(mv, dv, iv))
            cm[unsafe_offset=j] = mv
            cd[unsafe_offset=j] = dv
            ci[unsafe_offset=j] = iv
            j += 1
        var tm = pm
        pm = cm
        cm = tm
        var td = pd
        pd = cd
        cd = td
        var ti = pi
        pi = ci
        ci = ti
    if local:
        return best
    return _max3(
        pm[unsafe_offset=nq], pd[unsafe_offset=nq], pi[unsafe_offset=nq]
    )


def _alignment_trace(
    target: BPtr,
    query: BPtr,
    nt: Int,
    nq: Int,
    match_score: Float64,
    mismatch_score: Float64,
    open_gap_score: Float64,
    extend_gap_score: Float64,
    local: Bool,
    scores: FPtr,
    traces: BPtr,
    operations: BPtr,
    result: FPtr,
) -> Int:
    var cols = nq + 1
    var cells = (nt + 1) * cols
    var pm = scores
    var pd = scores.unsafe_offset(cols)
    var pi = scores.unsafe_offset(2 * cols)
    var cm = scores.unsafe_offset(3 * cols)
    var cd = scores.unsafe_offset(4 * cols)
    var ci = scores.unsafe_offset(5 * cols)
    var mt = traces
    var dt = traces.unsafe_offset(cells)
    var it = traces.unsafe_offset(2 * cells)

    if local:
        for j in range(cols):
            pm[unsafe_offset=j] = 0.0
            pd[unsafe_offset=j] = 0.0
            pi[unsafe_offset=j] = 0.0
    else:
        pm[unsafe_offset=0] = 0.0
        pd[unsafe_offset=0] = NEG
        pi[unsafe_offset=0] = NEG
        for j in range(1, nq + 1):
            pm[unsafe_offset=j] = NEG
            pd[unsafe_offset=j] = NEG
            pi[unsafe_offset=j] = (
                open_gap_score + Float64(j - 1) * extend_gap_score
            )
            it[unsafe_offset=j] = UInt8(0) if j == 1 else UInt8(2)

    var best = 0.0 if local else NEG
    var best_i = 0
    var best_j = 0
    var best_state = UInt8(0)
    for i in range(1, nt + 1):
        var row_start = i * cols
        if local:
            cm[unsafe_offset=0] = 0.0
            cd[unsafe_offset=0] = 0.0
            ci[unsafe_offset=0] = 0.0
        else:
            cm[unsafe_offset=0] = NEG
            cd[unsafe_offset=0] = (
                open_gap_score + Float64(i - 1) * extend_gap_score
            )
            ci[unsafe_offset=0] = NEG
            dt[unsafe_offset=row_start] = UInt8(0) if i == 1 else UInt8(1)
        var target_char = target[unsafe_offset=i - 1]
        for j in range(1, nq + 1):
            var idx = row_start + j
            var sub = (
                match_score if target_char
                == query[unsafe_offset=j - 1] else mismatch_score
            )

            var mbase = pm[unsafe_offset=j - 1]
            var ms = UInt8(0)
            if pd[unsafe_offset=j - 1] > mbase:
                mbase = pd[unsafe_offset=j - 1]
                ms = UInt8(1)
            if pi[unsafe_offset=j - 1] > mbase:
                mbase = pi[unsafe_offset=j - 1]
                ms = UInt8(2)
            var mv = mbase + sub
            var dv = pm[unsafe_offset=j] + open_gap_score
            var ds = UInt8(0)
            var candidate = pd[unsafe_offset=j] + extend_gap_score
            if candidate > dv:
                dv = candidate
                ds = UInt8(1)
            candidate = pi[unsafe_offset=j] + open_gap_score
            if candidate > dv:
                dv = candidate
                ds = UInt8(2)
            var iv = cm[unsafe_offset=j - 1] + open_gap_score
            var istate = UInt8(0)
            candidate = cd[unsafe_offset=j - 1] + open_gap_score
            if candidate > iv:
                iv = candidate
                istate = UInt8(1)
            candidate = ci[unsafe_offset=j - 1] + extend_gap_score
            if candidate > iv:
                iv = candidate
                istate = UInt8(2)
            if local:
                if mbase <= 0.0:
                    ms = UInt8(3)
                if mv <= 0.0:
                    mv = 0.0
                    ms = UInt8(3)
                if dv <= 0.0:
                    dv = 0.0
                    ds = UInt8(3)
                if iv <= 0.0:
                    iv = 0.0
                    istate = UInt8(3)
            cm[unsafe_offset=j] = mv
            cd[unsafe_offset=j] = dv
            ci[unsafe_offset=j] = iv
            mt[unsafe_offset=idx] = ms
            dt[unsafe_offset=idx] = ds
            it[unsafe_offset=idx] = istate
            if local:
                var state = _best_state(mv, dv, iv)
                var value = _max3(mv, dv, iv)
                if value > best:
                    best = value
                    best_i = i
                    best_j = j
                    best_state = state
        var tm = pm
        pm = cm
        cm = tm
        var td = pd
        pd = cd
        cd = td
        var ti = pi
        pi = ci
        ci = ti

    if not local:
        best_i = nt
        best_j = nq
        var idx = nt * cols + nq
        best_state = _best_state(
            pm[unsafe_offset=nq], pd[unsafe_offset=nq], pi[unsafe_offset=nq]
        )
        best = _max3(
            pm[unsafe_offset=nq], pd[unsafe_offset=nq], pi[unsafe_offset=nq]
        )

    result[unsafe_offset=0] = best
    result[unsafe_offset=1] = Float64(best_i)
    result[unsafe_offset=2] = Float64(best_j)
    var i = best_i
    var j = best_j
    var state = best_state
    var count = 0
    while state != UInt8(3):
        var idx = i * cols + j
        if state == UInt8(0):
            if i == 0 or j == 0:
                break
            operations[unsafe_offset=count] = UInt8(0)
            state = mt[unsafe_offset=idx]
            i -= 1
            j -= 1
        elif state == UInt8(1):
            if i == 0:
                break
            operations[unsafe_offset=count] = UInt8(1)
            state = dt[unsafe_offset=idx]
            i -= 1
        else:
            if j == 0:
                break
            operations[unsafe_offset=count] = UInt8(2)
            state = it[unsafe_offset=idx]
            j -= 1
        count += 1
    result[unsafe_offset=3] = Float64(i)
    result[unsafe_offset=4] = Float64(j)
    return count


def _line_end(data: BPtr, n: Int, start: Int) -> Int:
    var pos = start
    while pos < n and data[unsafe_offset=pos] != UInt8(10):
        pos += 1
    if pos > start and data[unsafe_offset=pos - 1] == UInt8(13):
        return pos - 1
    return pos


def _next_line(data: BPtr, n: Int, start: Int) -> Int:
    var pos = start
    while pos < n and data[unsafe_offset=pos] != UInt8(10):
        pos += 1
    return min(pos + 1, n)


def _fasta_count(data: BPtr, n: Int) -> Int:
    var count = 0
    var bol = True
    for i in range(n):
        if bol and data[unsafe_offset=i] == UInt8(62):
            count += 1
        bol = data[unsafe_offset=i] == UInt8(10)
    return count


def _fasta_scan(data: BPtr, n: Int, positions: IPtr) -> Int:
    var pos = 0
    var record = 0
    while pos < n:
        while pos < n and not (
            data[unsafe_offset=pos] == UInt8(62)
            and (pos == 0 or data[unsafe_offset=pos - 1] == UInt8(10))
        ):
            pos += 1
        if pos == n:
            break
        var header_start = pos + 1
        var header_end = _line_end(data, n, header_start)
        var seq_start = _next_line(data, n, header_start)
        pos = seq_start
        while pos < n and not (
            data[unsafe_offset=pos] == UInt8(62)
            and (pos == 0 or data[unsafe_offset=pos - 1] == UInt8(10))
        ):
            pos += 1
        positions[unsafe_offset=record * 4] = Int64(header_start)
        positions[unsafe_offset=record * 4 + 1] = Int64(header_end)
        positions[unsafe_offset=record * 4 + 2] = Int64(seq_start)
        positions[unsafe_offset=record * 4 + 3] = Int64(pos)
        record += 1
    return record


def _fastq_count(data: BPtr, n: Int) -> Int:
    var pos = 0
    var records = 0
    while pos < n:
        if data[unsafe_offset=pos] != UInt8(64):
            return -1
        pos = _next_line(data, n, pos)
        var seq_len = 0
        var found_plus = False
        while pos < n:
            if data[unsafe_offset=pos] == UInt8(43):
                found_plus = True
                pos = _next_line(data, n, pos)
                break
            var end = _line_end(data, n, pos)
            seq_len += end - pos
            pos = _next_line(data, n, pos)
        if not found_plus:
            return -1
        var quality_len = 0
        if seq_len == 0:
            if pos >= n:
                return -1
            pos = _next_line(data, n, pos)
        else:
            while pos < n and quality_len < seq_len:
                var end = _line_end(data, n, pos)
                quality_len += end - pos
                pos = _next_line(data, n, pos)
        if quality_len != seq_len:
            return -1
        records += 1
    return records


def _fastq_scan(data: BPtr, n: Int, positions: IPtr) -> Int:
    var pos = 0
    var record = 0
    while pos < n:
        if data[unsafe_offset=pos] != UInt8(64):
            return -1
        var header_start = pos + 1
        var header_end = _line_end(data, n, header_start)
        pos = _next_line(data, n, pos)
        var seq_start = pos
        var seq_len = 0
        var seq_end = pos
        var found_plus = False
        while pos < n:
            if data[unsafe_offset=pos] == UInt8(43):
                seq_end = pos
                found_plus = True
                pos = _next_line(data, n, pos)
                break
            var end = _line_end(data, n, pos)
            for k in range(pos, end):
                var c = data[unsafe_offset=k]
                if (
                    c == UInt8(32)
                    or c == UInt8(9)
                    or c == UInt8(11)
                    or c == UInt8(12)
                    or c == UInt8(13)
                ):
                    return -2
            seq_len += end - pos
            pos = _next_line(data, n, pos)
        if not found_plus:
            return -1
        var quality_start = pos
        var quality_end = pos
        var quality_len = 0
        if seq_len == 0:
            if pos >= n:
                return -1
            quality_end = _line_end(data, n, pos)
            for k in range(pos, quality_end):
                if data[unsafe_offset=k] < UInt8(33) or data[
                    unsafe_offset=k
                ] > UInt8(126):
                    return -3
            pos = _next_line(data, n, pos)
        else:
            while pos < n and quality_len < seq_len:
                quality_end = _line_end(data, n, pos)
                for k in range(pos, quality_end):
                    if data[unsafe_offset=k] < UInt8(33) or data[
                        unsafe_offset=k
                    ] > UInt8(126):
                        return -3
                quality_len += quality_end - pos
                pos = _next_line(data, n, pos)
        if quality_len != seq_len:
            return -1
        positions[unsafe_offset=record * 6] = Int64(header_start)
        positions[unsafe_offset=record * 6 + 1] = Int64(header_end)
        positions[unsafe_offset=record * 6 + 2] = Int64(seq_start)
        positions[unsafe_offset=record * 6 + 3] = Int64(seq_end)
        positions[unsafe_offset=record * 6 + 4] = Int64(quality_start)
        positions[unsafe_offset=record * 6 + 5] = Int64(quality_end)
        record += 1
    return record


def _complement(c: UInt8) -> UInt8:
    if c == UInt8(65):
        return UInt8(84)
    if c == UInt8(67):
        return UInt8(71)
    if c == UInt8(71):
        return UInt8(67)
    if c == UInt8(84) or c == UInt8(85):
        return UInt8(65)
    if c == UInt8(97):
        return UInt8(116)
    if c == UInt8(99):
        return UInt8(103)
    if c == UInt8(103):
        return UInt8(99)
    if c == UInt8(116) or c == UInt8(117):
        return UInt8(97)
    if c == UInt8(77):
        return UInt8(75)
    if c == UInt8(82):
        return UInt8(89)
    if c == UInt8(87):
        return UInt8(87)
    if c == UInt8(83):
        return UInt8(83)
    if c == UInt8(89):
        return UInt8(82)
    if c == UInt8(75):
        return UInt8(77)
    if c == UInt8(86):
        return UInt8(66)
    if c == UInt8(72):
        return UInt8(68)
    if c == UInt8(68):
        return UInt8(72)
    if c == UInt8(66):
        return UInt8(86)
    if c == UInt8(109):
        return UInt8(107)
    if c == UInt8(114):
        return UInt8(121)
    if c == UInt8(119):
        return UInt8(119)
    if c == UInt8(115):
        return UInt8(115)
    if c == UInt8(121):
        return UInt8(114)
    if c == UInt8(107):
        return UInt8(109)
    if c == UInt8(118):
        return UInt8(98)
    if c == UInt8(104):
        return UInt8(100)
    if c == UInt8(100):
        return UInt8(104)
    if c == UInt8(98):
        return UInt8(118)
    return c


def _complement_simd[
    W: Int
](chars: SIMD[DType.uint8, W],) -> SIMD[DType.uint8, W]:
    var result = chars
    result = chars.eq(UInt8(65)).select(SIMD[DType.uint8, W](84), result)
    result = chars.eq(UInt8(67)).select(SIMD[DType.uint8, W](71), result)
    result = chars.eq(UInt8(71)).select(SIMD[DType.uint8, W](67), result)
    result = (chars.eq(UInt8(84)) | chars.eq(UInt8(85))).select(
        SIMD[DType.uint8, W](65), result
    )
    result = chars.eq(UInt8(97)).select(SIMD[DType.uint8, W](116), result)
    result = chars.eq(UInt8(99)).select(SIMD[DType.uint8, W](103), result)
    result = chars.eq(UInt8(103)).select(SIMD[DType.uint8, W](99), result)
    result = (chars.eq(UInt8(116)) | chars.eq(UInt8(117))).select(
        SIMD[DType.uint8, W](97), result
    )
    result = chars.eq(UInt8(77)).select(SIMD[DType.uint8, W](75), result)
    result = chars.eq(UInt8(82)).select(SIMD[DType.uint8, W](89), result)
    result = chars.eq(UInt8(89)).select(SIMD[DType.uint8, W](82), result)
    result = chars.eq(UInt8(75)).select(SIMD[DType.uint8, W](77), result)
    result = chars.eq(UInt8(86)).select(SIMD[DType.uint8, W](66), result)
    result = chars.eq(UInt8(72)).select(SIMD[DType.uint8, W](68), result)
    result = chars.eq(UInt8(68)).select(SIMD[DType.uint8, W](72), result)
    result = chars.eq(UInt8(66)).select(SIMD[DType.uint8, W](86), result)
    result = chars.eq(UInt8(109)).select(SIMD[DType.uint8, W](107), result)
    result = chars.eq(UInt8(114)).select(SIMD[DType.uint8, W](121), result)
    result = chars.eq(UInt8(121)).select(SIMD[DType.uint8, W](114), result)
    result = chars.eq(UInt8(107)).select(SIMD[DType.uint8, W](109), result)
    result = chars.eq(UInt8(118)).select(SIMD[DType.uint8, W](98), result)
    result = chars.eq(UInt8(104)).select(SIMD[DType.uint8, W](100), result)
    result = chars.eq(UInt8(100)).select(SIMD[DType.uint8, W](104), result)
    result = chars.eq(UInt8(98)).select(SIMD[DType.uint8, W](118), result)
    return result


def _reverse_complement_range(
    src: BPtr,
    dst: BPtr,
    n: Int,
    start: Int,
    end: Int,
):
    comptime W = simd_width_of[DType.uint8]()
    var i = start
    while i + W <= end:
        var chars = src.unsafe_load[width=W](n - i - W).reversed()
        dst.unsafe_store(i, _complement_simd[W](chars))
        i += W
    while i < end:
        dst[unsafe_offset=i] = _complement(src[unsafe_offset=n - 1 - i])
        i += 1


@export("mbp_alignment_score")
def mbp_alignment_score(
    target_addr: Int,
    query_addr: Int,
    nt: Int,
    nq: Int,
    match_score: Float64,
    mismatch_score: Float64,
    open_gap_score: Float64,
    extend_gap_score: Float64,
    local: Int,
    work_addr: Int,
    work_capacity: Int,
) abi("C") -> Float64:
    if (
        target_addr == 0
        or query_addr == 0
        or work_addr == 0
        or nt <= 0
        or nq <= 0
        or nq > 1_000_000_000_000_000_000
        or work_capacity < 6 * (nq + 1)
    ):
        var zero = 0.0
        return zero / zero
    return _alignment_score(
        BPtr(unsafe_from_address=target_addr),
        BPtr(unsafe_from_address=query_addr),
        nt,
        nq,
        match_score,
        mismatch_score,
        open_gap_score,
        extend_gap_score,
        local != 0,
        FPtr(unsafe_from_address=work_addr),
    )


@export("mbp_alignment_trace")
def mbp_alignment_trace(
    target_addr: Int,
    query_addr: Int,
    nt: Int,
    nq: Int,
    match_score: Float64,
    mismatch_score: Float64,
    open_gap_score: Float64,
    extend_gap_score: Float64,
    local: Int,
    scores_addr: Int,
    traces_addr: Int,
    operations_addr: Int,
    result_addr: Int,
    scores_capacity: Int,
    traces_capacity: Int,
    operations_capacity: Int,
    result_capacity: Int,
) abi("C") -> Int:
    if (
        target_addr == 0
        or query_addr == 0
        or scores_addr == 0
        or traces_addr == 0
        or operations_addr == 0
        or result_addr == 0
        or nt <= 0
        or nq <= 0
        or nq > 1_000_000_000_000_000_000
        or scores_capacity < 6 * (nq + 1)
        or result_capacity < 5
    ):
        return -1
    var cols = nq + 1
    if (
        traces_capacity // 3 // cols < nt + 1
        or operations_capacity < nt
        or nq > operations_capacity - nt
    ):
        return -1
    return _alignment_trace(
        BPtr(unsafe_from_address=target_addr),
        BPtr(unsafe_from_address=query_addr),
        nt,
        nq,
        match_score,
        mismatch_score,
        open_gap_score,
        extend_gap_score,
        local != 0,
        FPtr(unsafe_from_address=scores_addr),
        BPtr(unsafe_from_address=traces_addr),
        BPtr(unsafe_from_address=operations_addr),
        FPtr(unsafe_from_address=result_addr),
    )


@export("mbp_fasta_scan")
def mbp_fasta_scan(
    data_addr: Int, n: Int, positions_addr: Int, positions_capacity: Int
) abi("C") -> Int:
    if (
        data_addr == 0
        or positions_addr == 0
        or n <= 0
        or positions_capacity <= 0
    ):
        return -1
    var needed = _fasta_count(BPtr(unsafe_from_address=data_addr), n)
    if needed > positions_capacity:
        return -1
    return _fasta_scan(
        BPtr(unsafe_from_address=data_addr),
        n,
        IPtr(unsafe_from_address=positions_addr),
    )


@export("mbp_fastq_scan")
def mbp_fastq_scan(
    data_addr: Int, n: Int, positions_addr: Int, positions_capacity: Int
) abi("C") -> Int:
    if (
        data_addr == 0
        or positions_addr == 0
        or n <= 0
        or positions_capacity <= 0
    ):
        return -1
    var needed = _fastq_count(BPtr(unsafe_from_address=data_addr), n)
    if needed < 0 or needed > positions_capacity:
        return -1
    return _fastq_scan(
        BPtr(unsafe_from_address=data_addr),
        n,
        IPtr(unsafe_from_address=positions_addr),
    )


@export("mbp_reverse_complement")
def mbp_reverse_complement(
    src_addr: Int, n: Int, dst_addr: Int, dst_capacity: Int
) abi("C") -> Int:
    if n < 0 or dst_capacity < n:
        return -1
    if n == 0:
        return 0
    if src_addr == 0 or dst_addr == 0:
        return -1
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    _reverse_complement_range(src, dst, n, 0, n)
    return 0
