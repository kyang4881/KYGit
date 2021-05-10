<!-- if you see the script, make sure to use path: http://localhost/kyPHPproject/firstwebpage/test-site/index.php -->
<?php
// Initialize session
session_start();
?>

<!DOCTYPE html>
<html class="indexhome" lang="en" dir="ltr">
  <head>
    <meta charset="utf-8" name = "viewport" content="width=device-width, initial-scale=1.0">
    <title>KY</title>
  </head>
    <link rel="stylesheet" href="styles/webstyles.css">

  <body class="indexbg">

    <div class="topnav">
      <input type="text" placeholder="">
      <div id="topnav_mag"> <input type="image" src="images/magnifying.png" alt="Search"></div>
    </div>

    <!--<div class="bg2" style="background-image: url('');">-->

    <div id="login1">
      <a href="scripts/login.php">Login</a>
    </div>

    <div>
      <ul class="nav1_ul">
        <li class="dropdown1">
          <!-- <a href="scripts/loggedin.php">My Account | </a> -->
          <a href="javascript:void(0)" class="dropbtn">My Account</a>
          <div class="index_acc_dropdown">
            <a href="scripts/loggedin.php">Account Setting</a>
            <a href="#">Order History</a>
            <a href="#">Manage Payment</a>
          </div>
        </li>
        <!--<li><a href="index.php">Gift Cards</a></li>
        <li><a href="index.php">Deals Store</a></li>
        <li><a href="index.php">Best Sellers</a></li>
        <li><a href="index.php">Buy Again</a></li>
        <li><a href="index.php">Membership</a></li>
        <li><a href="index.php">Store</a></li>
        <li><a href="index.php">Subsribe & Save</a></li>
        <li><a href="index.php">Gift Ideas</a></li>
        <li><a href="index.php">Customer Service</a></li>
        <li><a href="index.php">Holiday Deals</a></li>-->
      </ul>
    </div>

    <?php
    require('C:/Users/kyyan/Documents/xampp/htdocs/kyPHPproject/firstwebpage/test-site/scripts/connection.php');

    $sql = "SELECT ask_price_final,
                   sold_price_final,
                   city_final,
                    listed_in_days_final,
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
                    sigma_est_rental_dom_final
          FROM testschema.house_sigma_data limit 1";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {

      echo "<br><br><br>
            <table id='t1'>
              <caption>From the Database: testschema.house_sigma_data</caption>
              <tr>
                <th>ask_price_final</th>
                <th>sold_price_final</th>
                <th>city_final</th>
                <th>listed_in_days_final</th>
                <th>listed_in_date_final</th>
                <th>sold_in_days_final</th>
                <th>sold_in_date_final</th>
                <th>bedroom_final</th>
                <th>bathroom_final</th>
                <th>garage</th>
                <th>property_tax_final</th>
                <th>building_type_final</th>
                <th>building_age_final</th>
                <th>building_size_final</th>
                <th>lot_size_final</th>
                <th>parking_final</th>
                <th>basement_final</th>
                <th>mls_number_final</th>
                <th>days_on_mrkt_final</th>
                <th>listed_date_final</th>
                <th>updated_date_final</th>
                <th>sigma_est_price_final</th>
                <th>sigma_est_date_final</th>
                <th>sigma_est_rent_final</th>
                <th>sigma_est_rental_yield_final</th>
                <th>sigma_est_rental_dom_final</th>
              </tr>";
      // output data of each row
      while($row = $result->fetch_assoc()) {
        echo "<tr>
                <td>".$row["ask_price_final"]."</td>
                <td>".$row["sold_price_final"]."</td>
                <td>".$row["city_final"]."</td>
                <td>".$row["listed_in_days_final"]."</td>
                <td>".$row["listed_in_date_final"]."</td>
                <td>".$row["sold_in_days_final"]."</td>
                <td>".$row["sold_in_date_final"]."</td>
                <td>".$row["bedroom_final"]."</td>
                <td>".$row["bathroom_final"]."</td>
                <td>".$row["garage"]."</td>
                <td>".$row["property_tax_final"]."</td>
                <td>".$row["building_type_final"]."</td>
                <td>".$row["building_age_final"]."</td>
                <td>".$row["building_size_final"]."</td>
                <td>".$row["lot_size_final"]."</td>
                <td>".$row["parking_final"]."</td>
                <td>".$row["basement_final"]."</td>
                <td>".$row["mls_number_final"]."</td>
                <td>".$row["days_on_mrkt_final"]."</td>
                <td>".$row["listed_date_final"]."</td>
                <td>".$row["updated_date_final"]."</td>
                <td>".$row["sigma_est_price_final"]."</td>
                <td>".$row["sigma_est_date_final"]."</td>
                <td>".$row["sigma_est_rent_final"]."</td>
                <td>".$row["sigma_est_rental_yield_final"]."</td>
                <td>".$row["sigma_est_rental_dom_final"]."</td>
              </tr>";
      }
      echo "</table>";
    } else {
      echo "0 results";
    }
    $conn->close();

    ?><br>

    <?php //require('scripts/db_tables.php') ?>


  </body>
</html>
