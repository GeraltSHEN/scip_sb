# SCIP Strong Branching: How `up/down` are computed for infeasible or cutoff children

This note summarizes how SCIP computes strong-branching `down` and `up` values in:

- `src/scip/branch_vanillafullstrong.c`
- `src/scip/scip_var.c`
- `src/scip/lp.c`

The key point is: for infeasible/cutoff children, `down/up` are treated as **dual bounds capped by the global cutoff bound**, not necessarily the true child LP optimum.

## 1) Entry point in `branch_vanillafullstrong.c`

In the candidate loop, values are initialized to `-infinity` as sentinels, then filled by strong branching:

```c
up = -SCIPinfinity(scip);
down = -SCIPinfinity(scip);

if( integral )
{
   SCIP_CALL( SCIPgetVarStrongbranchInt(scip, cands[c], INT_MAX, idempotent,
         &down, &up, &downvalid, &upvalid, &downinf, &upinf, &downconflict, &upconflict, &lperror) );
}
else
{
   SCIP_CALL( SCIPgetVarStrongbranchFrac(scip, cands[c], INT_MAX, idempotent,
         &down, &up, &downvalid, &upvalid, &downinf, &upinf, &downconflict, &upconflict, &lperror) );
}
```

After the call, the rule uses:

```c
down = MAX(down, lpobjval);
up = MAX(up, lpobjval);
downgain = down - lpobjval;
upgain = up - lpobjval;
```

So branch gains are always measured relative to the current LP objective.

## 2) Wrapper behavior in `scip_var.c`

`SCIPgetVarStrongbranchFrac()` / `SCIPgetVarStrongbranchInt()` call:

```c
SCIPcolGetStrongbranch(..., &localdown, &localup, &localdownvalid, &localupvalid, lperror);
```

Then they classify cutoff/infeasible sides via cutoff bound comparison:

```c
*downinf = localdownvalid && SCIPsetIsGE(scip->set, localdown, scip->lp->cutoffbound);
*upinf   = localupvalid   && SCIPsetIsGE(scip->set, localup,   scip->lp->cutoffbound);
```

This means `downinf/upinf` are "cutoff w.r.t. cutoffbound (and valid)," not just a raw LP status label.

## 3) Core computation in `lp.c` (`SCIPcolGetStrongbranch`)

### 3.1 Call to LPI strong branching

SCIP starts from column-only LP objective values:

```c
sbdown = lp->lpobjval;
sbup = lp->lpobjval;

retcode = SCIPlpiStrongbranchInt/Frac(..., &sbdown, &sbup, &sbdownvalid, &sbupvalid, ...);
```

### 3.2 Postprocessing to SCIP-side bound

On success, SCIP converts to full objective and clips by global cutoff bound:

```c
looseobjval = getFiniteLooseObjval(lp, set, prob);
sbdown = MIN(sbdown + looseobjval, lp->cutoffbound);
sbup = MIN(sbup + looseobjval, lp->cutoffbound);
```

So if a child is infeasible/cutoff, the final stored value is typically at/near `lp->cutoffbound`.

### 3.3 Important edge cases

- If `retcode == SCIP_LPERROR`, values are set invalid and not trusted.
- If `lp->looseobjvalinf > 0`, SCIP sets `sbdown/sbup = -infinity` with invalid flags (no usable gain).

## 4) Practical interpretation for infeasible/cutoff children

For a successful strong-branch call with valid dual bound:

1. LPI returns a side bound (`raw_sb`).
2. SCIP shifts by `looseobjval`.
3. SCIP clips by `cutoffbound`.
4. `scip_var.c` sets `downinf/upinf` if valid and `>= cutoffbound`.

So in most infeasible/cutoff cases, the side value used by `vanillafullstrong` is effectively the cutoff bound, and the score uses gains computed from that capped value.
