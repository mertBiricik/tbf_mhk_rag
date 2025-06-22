

    var today = new Date();
    var dd = today.getDate();

    var mm = today.getMonth() + 1;
    var yyyy = today.getFullYear();
    if (dd < 10) {
        dd = '0' + dd;
    }

    if (mm < 10) {
        mm = '0' + mm;
    }
    today = dd + '/' + mm + '/' + yyyy;




    var $DatatableForEvents;
    function DataDoldur() {
        var _DatatableForEvents = $('#TableMaclar');
        $DatatableForEvents = _DatatableForEvents
            .DataTable({
                //dom: 'Bfrtip',
                //buttons: [
                //    'copy', 'csv', 'excel', 'pdf', 'print'
                //],
                "ordering": false, //true

		
                "order": [[0, "asc"], [7, "asc"]]
		
            });



        var $DataTableButtonsForGames = new $.fn.dataTable.Buttons($DatatableForEvents, {
            buttons: [{
                extend: 'excelHtml5',
                className: 'transparent-button float-right btn bg-c-excel-icon text-c-white my-2 mx-1',
                text: '<i class=""fa fa-file-excel-o rounded mr-2""></i> Excel\'e Aktar',
                exportOptions: {
                    stripHtml: true,
                    stripNewlines: true,
                    columns: 'th:not(.not-export)',
                    format: {
                        body: function (data, row, column, node) {
                            return data.replace(/<br\s*\/?>/ig, ' | ').replace(/(<([^>]+)>)/ig, '');
                        }
                    }
                },
            },
            {
                extend: 'print',
                autoPrint: false,
                className: 'transparent-button float-right d-none d-lg-block btn bg-c-league text-c-white my-2 mx-1',
                text: '<i class=""fa fa-clone rounded mr-2""></i>Kopyala',
                orientation: 'landscape',
                pageSize: 'LEGAL',
                exportOptions: {
                    stripHtml: true,
                    stripNewlines: true,
                    columns: 'th:not(.not-export)',
                    format: {
                        body: function (data, row, column, node) {
                            return data.replace(/<br\s*\/?>/ig, '___').replace(/(<([^>]+)>)/ig, '').replace('___', '<br />');
                        }
                    }
                },
            }]
        }).container().appendTo($('.dt-buttons-container'));
        // $('#TableMaclar').DataTable( {
        //       dom: 'Bfrtip',
        //       buttons: [
        //           'copy', 'csv', 'excel', 'pdf', 'print'
        //       ]
        //   } );

        //var $DataTableButtonsForGames = new $.fn.dataTable.Buttons($DatatableForEvents, {
        //                           buttons: [{
        //                               extend: 'excelHtml5',
        //                               className: 'transparent-button float-right btn bg-c-excel-icon text-c-white my-2 mx-1',
        //                               text: '<i class=""fa fa-file-excel-o rounded mr-2""></i> Excel\'e Aktar',
        //                               exportOptions: {
        //                                   stripHtml: true,
        //                                   stripNewlines: true,
        //                                   columns: 'th:not(.not-export)',
        //                                   format: {
        //                                       body: function (data, row, column, node) {
        //                                           return data.replace(/<br\s*\/?>/ig, ' | ').replace(/(<([^>]+)>)/ig, '');
        //                                       }
        //                                   }
        //                               },
        //                           },
        //                           {
        //                               extend: 'print',
        //                               autoPrint: false,
        //                               className: 'transparent-button float-right d-none d-lg-block btn bg-c-league text-c-white my-2 mx-1',
        //                               text: '<i class=""fa fa-clone rounded mr-2""></i>Kopyala',
        //                               orientation: 'landscape',
        //                               pageSize: 'LEGAL',
        //                               exportOptions: {
        //                                   stripHtml: true,
        //                                   stripNewlines: true,
        //                                   columns: 'th:not(.not-export)',
        //                                   format: {
        //                                       body: function (data, row, column, node) {
        //                                           return data.replace(/<br\s*\/?>/ig, '___').replace(/(<([^>]+)>)/ig, '').replace('___', '<br />');
        //                                       }
        //                                   }
        //                               },
        //                           }]
        //                       }).container().appendTo($('.dt-buttons-container'));

        yadcf.init($DatatableForEvents,
            [
                {
                    column_number: 0,
                    filter_type: "range_date",
                    datepicker_type: 'bootstrap-datetimepicker',
                    date_format: 'DD.MM.YYYY',
                    filter_plugin_options: $DatepickerDefaults,
                    filter_container_id: "FilterContainer_GameDate",
                    filter_default_label: ["../../....", "../../...."],
                    style_class: "custom-form"
                },

                {
                    column_number: 3,

                    filter_default_label: "Hafta Seç",
                    filter_container_id: "FilterContainer_GameWeek",
                    style_class: "custom-select",
                    filter_match_mode: "exact"

                },
                {
                    column_number: 4,
                    filter_default_label: "A Takım Seç",
                    filter_container_id: "FilterContainer_Home",
                    style_class: "custom-select"
                },
                {
                    column_number: 6,
                    filter_default_label: "B Takım Seç",
                    filter_container_id: "FilterContainer_Away",
                    style_class: "custom-select"
                },
                {
                    column_number: 7,
                    filter_default_label: "Grup Seç",
                    filter_container_id: "FilterContainer_Group",
                    style_class: "custom-select",
                    filter_match_mode: "exact"
                },
                {
                    column_number: 8,
                    filter_default_label: "Şehir Seç",
                    filter_container_id: "FilterContainer_City",
                    style_class: "custom-select"

                },
                {
                    column_number: 9,
                    filter_default_label: "Salon Seç",
                    filter_container_id: "FilterContainer_Hall",
                    style_class: "custom-select"
                },
                {
                    column_number: 10,
                    filter_default_label: "Yayın Seç",
                    filter_container_id: "FilterContainer_Broadcast",
                    column_data_type: "html",
                    filter_match_mode: "contains",
                    html_data_type: "text",
                    style_class: "custom-select"
                }
            ]);


        yadcf.exFilterColumn(_DatatableForEvents, [
            [0, [today]],
            [3, ["03. Hafta"]]
        ]);

    }
