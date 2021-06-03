=======
Openapi
=======

`django-rest-framework`_ makes it easy to automatically create online schema documentation. Here are the key lines of code from :code:`config.urls`. :func:`get_schema_view` creates the schema, which gets served a boilerplate template at the :code:`swagger-ui` endpoint and you end up with informative, functional and good looking documentation of your schema. Openapi is now the de facto standard for documenting APIs and as such rich tooling exists, including `openapi-generator <https://github.com/OpenAPITools/openapi-generator>`_, which can be used to autogenerate a client library.

::

    urlpatterns = [
        path("grants/", include("grants.urls")),
        path("admin/", admin.site.urls),
        path(
            "openapi/",
            get_schema_view(
                title="Grants Admin API", description="Grants Admin API", version="0.1.1"
            ),
            name="openapi-schema",
        ),
        path(
            "swagger-ui/",
            TemplateView.as_view(
                template_name="swagger-ui.html",
                extra_context={"schema_url": "openapi-schema"},
            ),
            name="swagger-ui",
        ),
    ]



Schemas
-------

..  image:: _static/openapi_summary.png
    :alt: Open API Schemas


End Points
-------

..  image:: _static/openapi_endpoints.png
    :alt: Open API Endpoints

End Point Detail
-------

..  image:: _static/openapi_endpoint_detail.png
    :alt: Open API Endpoint Detail


..  _django-rest-framework: https://www.django-rest-framework.org