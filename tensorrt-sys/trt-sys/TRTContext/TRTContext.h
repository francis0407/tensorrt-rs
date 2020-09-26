#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-deprecated-headers"
//
// Created by mason on 9/17/19.
//

#ifndef LIBTRT_TRTCONTEXT_H
#define LIBTRT_TRTCONTEXT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Context;
typedef struct Context Context_t;

void destroy_excecution_context(Context_t* execution_context);

void context_set_name(Context_t* execution_context, const char *name);
const char* context_get_name(Context_t *execution_context);

void execute(const Context_t* execution_context, const void** binding_data, const int num_bindings, const size_t* data_sizes);

#ifdef __cplusplus
};
#endif


#endif //LIBTRT_TRTCONTEXT_H

#pragma clang diagnostic pop