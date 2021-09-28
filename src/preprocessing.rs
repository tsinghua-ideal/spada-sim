use std::collections::HashMap;

use priority_queue::PriorityQueue;
use sprs::vec;

use crate::storage::CsrMatStorage;
use crate::trace_println;

pub type RowMap = HashMap<usize, usize>;

pub fn affinity_based_row_reordering(
    amat: &mut CsrMatStorage,
    cache_size: usize,
    a_avg_row_len: usize,
    b_avg_row_len: usize,
) -> Option<RowMap> {
    println!("---Affinity based row reordering---");
    // Calculate the window size.
    let w = cache_size / (a_avg_row_len * b_avg_row_len);
    let mut pq = PriorityQueue::new();
    let mut rowmap = RowMap::new();

    let mut rows = vec![];
    for rowid in 0..amat.indptr.len() - 1 {
        if amat.indptr[rowid + 1] >= amat.indptr[rowid] {
            rows.push(rowid);
            pq.push(rowid, 0);
        }
    }

    if rows.len() <= 0 {
        return None;
    } else {
        rowmap.insert(0, 0);
    }

    for i in 0..rows.len() {
        println!("reorder row {}", &i);
        for u in amat.read_row(rowmap[&i]).unwrap().indptr.iter() {
            for r in find_contain_rows(amat, *u).iter() {
                if let Some((item, priority)) = pq.get(r).map(|(x, y)| (x.clone(), y.clone())) {
                    pq.change_priority(&item, priority + 1).unwrap();
                }
            }
        }

        if i >= w {
            for u in amat.read_row(rowmap[&(i - w)]).unwrap().indptr.iter() {
                for r in find_contain_rows(amat, *u).iter() {
                    if let Some((item, priority)) = pq.get(r).map(|(x, y)| (x.clone(), y.clone())) {
                        pq.change_priority(&item, priority - 1).unwrap();
                    }
                }
            }
        }

        // rowmap[&i] = pq.pop().unwrap().0;
        // *rowmap.get_mut(&(i+1)).unwrap() = pq.pop().unwrap().0;
        rowmap.insert(i + 1, pq.pop().unwrap().0);
    }

    Some(rowmap)
}

fn find_contain_rows(amat: &CsrMatStorage, colid: usize) -> Vec<usize> {
    let mut result = vec![];
    for ipid in 0..amat.indptr.len() - 1 {
        if amat.indices[amat.indptr[ipid]..amat.indptr[ipid + 1]].contains(&colid) {
            result.push(ipid);
        }
    }

    result
}

pub fn sort_by_length(amat: &mut CsrMatStorage) -> HashMap<usize, usize> {
    trace_println!("---Sort A rows by length---");
    let mut id_len_vector = vec![];
    for idx in 0..amat.row_num() {
        id_len_vector.push([idx, amat.rowptr(idx + 1) - amat.rowptr(idx)]);
    }
    id_len_vector.sort_by(|a, b| a[1].cmp(&b[1]));
    let remap: HashMap<usize, usize> = id_len_vector
        .iter()
        .enumerate()
        .map(|x| (x.0, x.1[0]))
        .collect();
    return remap;
}
