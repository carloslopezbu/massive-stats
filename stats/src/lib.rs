use pyo3::prelude::*;
use rayon::prelude::*;

/// Calcula la media de cada fila (subvector) en paralelo
#[pyfunction]
fn medias_por_fila(matriz: Vec<Vec<f64>>) -> Vec<f64> {
    matriz
        .par_iter()
        .map(|fila| {
            if fila.is_empty() {
                0.0
            } else {
                fila.iter().sum::<f64>() / fila.len() as f64
            }
        })
        .collect()
}

/// Módulo Python
#[pymodule]
fn stats(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(medias_por_fila, m)?)?;  // Changed to add_function
    Ok(())
}