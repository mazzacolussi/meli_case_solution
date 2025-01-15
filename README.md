Case de Logística - Meli
==================================================

Neste case foi proposto a escolha de 1 entre 4 problemáticas. Sendo assim, optou-se pela primeira proposta:

"""
En la vertical de pricing están interesados en dar sugerencias de descuentos para los
ítems del Marketplace. Actualmente, tienen un equipo experto en jación de precios,
los cuales revisan ítems manualmente para encontrar si el ítem requiere un descuento
y cuál sería el descuento adecuado para generar un aumento en las ventas en el
corto plazo. Este equipo busca disminuir las cargas manuales que tienen sus
colaboradores para que puedan dedicarse a otras actividades más rentables para el
negocio.
"""


# Etapas de desenvolvimento

Para garantir o bom funcionamento do desenvolvimento, recomenda-se que instale os pacotes necessário:

```bash
pip install -r requirements.txt
```


## 1. Geração da base raw

A primeira etapa do desenvolvimento consiste na amostragem via API fornecida neste case. Para isso, foi iterado para cada `site_id` (`MLA`, `MLB`, etc) existente em que a resposta (`result`) não fosse vazia, os arquivos de apoio se encontram na pasta [src/config](src/config). Também, como proposto, iterou-se sobre os offsets discretizados (offset $\in \{0, 50, 100, ..., 950\}$) e para todas as categorias de produtos existentes no marketplace da respectiva região da iteração.

```bash
python src/data/download_raw_data.py
```