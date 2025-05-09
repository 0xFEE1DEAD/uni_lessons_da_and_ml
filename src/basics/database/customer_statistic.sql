with customer_statistic as (
	select
		c.customer_id,
		c.first_name,
		c.last_name,
		sum(p.amount) as sum_of_payments,
		count(r.rental_id) as count_of_rents,
		avg(p.amount) as avg_payment,
		case -- Хотя бы один CASE
			when max(r.rental_date) > '2005-08-23'::date then 'active'
			else 'inactive'
		end as is_active
	from customer c 
	join rental r on r.customer_id = c.customer_id
	join payment p on p.rental_id  = r.rental_id -- хотя бы 2 join
	group by c.customer_id -- хотя бы одно использование group by
	having sum(p.amount) > 0
)
select
 	*,
 	rank() over (order by cs.sum_of_payments DESC) as rank_by_sum_of_payments -- Хотя бы одна оконная функция
from customer_statistic cs
order by rank_by_sum_of_payments