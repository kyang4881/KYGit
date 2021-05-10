<?php
require('connection.php');

$sql = "SELECT ask_price_final,
               sold_price_final,
               city_final
                /*listed_in_days_final,
                listed_in_date_final,
                sold_in_days_final,
                sold_in_date_final,
                bedroom_final,
                bathroom_final,
                garage,
                property_tax_final,
                building_type_final,
                building_age_final,
                building_size_final,
                lot_size_final,
                parking_final,
                basement_final,
                mls_number_final,
                days_on_mrkt_final,
                listed_date_final,
                updated_date_final,
                sigma_est_price_final,
                sigma_est_date_final,
                sigma_est_rent_final,
                sigma_est_rental_yield_final,
                sigma_est_rental_dom_final*/
      FROM testschema.house_sigma_data";
$result = $conn->query($sql);

if ($result->num_rows > 0) {

  echo "<br><br><br>
        <table id='t1'>
          <caption>From the Database: testschema.house_sigma_data</caption>
          <tr>
            <th>ask_price_final</th>
            <th>sold_price_final</th>
            <th>city_final</th>
          </tr>";
  // output data of each row
  while($row = $result->fetch_assoc()) {
    echo "<tr>
            <td>".$row["ask_price_final"]."</td>
            <td>".$row["sold_price_final"]."</td>
            <td>".$row["city_final"]."</td>
          </tr>";
  }
  echo "</table>";
} else {
  echo "0 results";
}
$conn->close();

?><br>
